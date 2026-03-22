import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import scipy.linalg as la
# ==========================================
def build_block_diagonal_sigma(d_var, block_size, rho_intra=0.8):
    """构造分块对角相关矩阵"""
    num_blocks = d_var // block_size
    blocks = []
    for _ in range(num_blocks):
        # 组内相关系数为 rho_intra，对角线为 1
        block = np.full((block_size, block_size), rho_intra)
        np.fill_diagonal(block, 1.0)
        blocks.append(block)
    
    # 组装成分块对角矩阵
    R_true = la.block_diag(*blocks)
    
    # 将相关矩阵转化为协方差矩阵 (这里假设各个变量的无条件方差为 1)
    Sigma_true = R_true 
    return Sigma_true, R_true

def build_toeplitz_sigma(d_var, rho=0.7):
    """构造 AR(1) 空间衰减相关矩阵"""
    # 生成第一行 [1, rho, rho^2, rho^3, ...]
    first_row = rho ** np.arange(d_var)
    
    # 利用 scipy 生成 Toeplitz 矩阵
    R_true = la.toeplitz(first_row)
    Sigma_true = R_true
    return Sigma_true, R_true

def build_equicorrelation_sigma(d_var, rho=0.5):
    """构造等相关矩阵"""
    R_true = np.full((d_var, d_var), rho)
    np.fill_diagonal(R_true, 1.0)
    Sigma_true = R_true
    return Sigma_true, R_true
# 1. 模拟高维多元时间序列生成器
# ==========================================
def generate_var1_data(num_samples=5000, d_var=10, spectral_radius=0.8, seed=42,structure='toeplitz'):
    """
    生成一个 D 维的 VAR(1) 过程: X_t = A * X_{t-1} + epsilon_t
    epsilon_t ~ N(0, Sigma)
    """
    np.random.seed(seed)
    
    # 1. 生成转移矩阵 A (控制时间维度的自回归特性)
    A = np.random.randn(d_var, d_var)
    # 缩放 A 的特征值以保证系统的平稳性 (Stationarity)
    # 最大特征值的绝对值（谱半径）必须小于 1
    eigenvalues = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigenvalues))
    A = A / max_eig * spectral_radius
    
    # 2. 核心修改：使用具有明确统计结构的真实协方差矩阵
    if structure == "block":
        # 假设 d_var=8, 分为 2 个 4x4 的强相关区块
        Sigma_true, R_true = build_block_diagonal_sigma(d_var, block_size=4, rho_intra=0.85)
    elif structure == "toeplitz":
        Sigma_true, R_true = build_toeplitz_sigma(d_var, rho=0.8)
    elif structure == "equi":
        Sigma_true, R_true = build_equicorrelation_sigma(d_var, rho=0.6)
    else:
        raise ValueError("未知的协方差结构")
    # 2. 生成真实的协方差矩阵 Sigma (控制截面维度的高度相关性，用来测试 Copula)

    # 为了保证半正定，我们先生成一个下三角矩阵 L
    #L_true = np.tril(np.random.randn(d_var, d_var))
    # 增加对角线元素的值，确保矩阵良态（Well-conditioned）
    #np.fill_diagonal(L_true, np.abs(np.diagonal(L_true)) + 1.0)
    #Sigma_true = L_true @ L_true.T
    
    # 将 Sigma_true 转化为真实的相关系数矩阵 R_true (方便后续与模型预测的 R 对比)
    #inv_std = np.diag(1.0 / np.sqrt(np.diag(Sigma_true)))
    #R_true = inv_std @ Sigma_true @ inv_std
    
    print("理论相关系数矩阵 R_true 的条件数:", np.linalg.cond(R_true))
    
    # 3. 迭代生成时间序列数据
    X = np.zeros((num_samples, d_var))
    # 使用多元高斯分布生成噪声
    epsilon = np.random.multivariate_normal(mean=np.zeros(d_var), cov=Sigma_true, size=num_samples)
    
    for t in range(1, num_samples):
        X[t] = A @ X[t-1] + epsilon[t]

        # 加上多变量的混合正弦周期项 (周期为 4 和 96)
        X[t] += np.sin(2 * np.pi * t / 4) * 0.5 * np.random.randn(d_var)
        X[t] += np.cos(2 * np.pi * t / 96) * 0.3
        
    return X, R_true,A, Sigma_true

# ==========================================
# 2. 数据清洗与标准化处理
# ==========================================
class TimeSeriesPreprocessor:
    def __init__(self):
        # 使用 StandardScaler 进行 Z-score 标准化，这对于 NLL Loss 的稳定性极其重要
        self.scaler = StandardScaler()
        
    def fit_transform(self, data):
        # 严格来说，只能在训练集上 fit，避免未来信息泄露
        return self.scaler.fit_transform(data)
        
    def transform(self, data):
        return self.scaler.transform(data)
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# ==========================================
# 3. PyTorch Dataset 构建
# ==========================================
class TimeSeriesDataset(Dataset):
    """
    使用滑动窗口 (Sliding Window) 截取时序数据
    """
    def __init__(self, data, seq_len, pred_len):
        """
        data: 标准化后的 numpy 数组 [Total_Length, D]
        seq_len: 输入给 LLM 的历史长度 (T)
        pred_len: 需要预测的未来长度 (H)
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 计算可以截取出多少个样本
        self.num_samples = len(data) - seq_len - pred_len + 1
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # X: [seq_len, D]
        x_start = idx
        x_end = idx + self.seq_len
        x = self.data[x_start : x_end]
        
        # Y: [pred_len, D]
        y_start = x_end
        y_end = x_end + self.pred_len
        y = self.data[y_start : y_end]
        
        return x, y