import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 导入你之前写的评估指标 (确保 evaluator.py 在同级目录)
from metrics.evaluator import calculate_point_metrics, energy_score, gaussian_analytical_crps

@torch.no_grad()
def evaluate_copula_model(model, test_loader, device="cuda", num_samples=100):
    """
    全维度评估管线 (适配真实数据集与无 R_true 的场景)
    加入自动嗅探机制，兼容高维静态 Copula 与低维动态 Copula
    """
    model.eval()
    model.to(device)
    
    total_mse, total_mae, total_crps, total_es = 0.0, 0.0, 0.0, 0.0
    num_batches = 0
    
    all_R_pred = []
    all_y_true_flat = [] # 用于计算经验相关性矩阵
    
    print("🚀 开始在测试集上进行全维度评估...")
    progress_bar = tqdm(test_loader, desc="Evaluating")
    
    for batch_x, batch_y in progress_bar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
        B, T, D = batch_y.shape
        
        # 收集真实的 y 用于后续计算经验相关系数
        # 取每个样本预测窗口的最后一个时间步
        all_y_true_flat.append(batch_y[:, -1, :].cpu().numpy()) 
        
        # 1. 前向传播 (混合精度加速)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            mu, sigma, L_unnormalized = model(batch_x)
            
        # 强制转为 float32 保障统计精度
        mu, sigma, L_unnormalized = mu.float(), sigma.float(), L_unnormalized.float()
        
        # =========================================================
        # 【自适应修复 1】：嗅探当前是动态还是静态 Copula
        # =========================================================
        is_dynamic = (L_unnormalized.dim() == 4)
        
        # ---------------------------------------------------------
        # 2. 重构相关系数矩阵 (维度自适应松绑版)
        # ---------------------------------------------------------
        L_tril = torch.tril(L_unnormalized)
        
        # 【自适应修复 2】：动态构建对角线掩码
        if is_dynamic:
            diag_mask = torch.eye(D, device=device, dtype=torch.bool).expand(B, T, D, D)
        else:
            diag_mask = torch.eye(D, device=device, dtype=torch.bool).expand(B, D, D)
            
        # 使用 1e-5 极小值安全垫
        diag_elements = F.softplus(torch.diagonal(L_tril, dim1=-2, dim2=-1)) + 1e-5
        L_pd = L_tril * (~diag_mask) + torch.diag_embed(diag_elements)
        
        # 行范数归一化，得到纯粹的 Cholesky 因子
        row_norms = torch.norm(L_pd, p=2, dim=-1, keepdim=True)
        L_normalized = L_pd / (row_norms + 1e-8)
        
        # 【自适应修复 3】：提取用于画图的热力图矩阵 R_pred
        if is_dynamic:
            # 低维动态：提取预测窗口的最后一个时间步
            L_last = L_normalized[:, -1, :, :] 
        else:
            # 高维静态：整个窗口共享同一个结构，直接使用
            L_last = L_normalized 
            
        # 计算相关性矩阵 R = L @ L.T
        R_pred = torch.einsum('bdi, bji -> bdj', L_last, L_last)
        all_R_pred.append(R_pred.mean(dim=0).cpu().numpy())
        
        # ---------------------------------------------------------
        # 3. 极速省显存版：手动重参数化蒙特卡洛采样 (自适应 Einsum)
        # 彻底绕开 PyTorch MultivariateNormal 的显存黑洞
        # ---------------------------------------------------------
        
        # (1) 直接在 GPU 上生成 100 次独立的标准正态噪声 Z ~ N(0, I)
        z_standard = torch.randn(num_samples, B, T, D, device=device, dtype=L_normalized.dtype)
        
        # (2) 用 einsum 高效计算矩阵乘法: L_normalized * Z
        # 【自适应修复 4】：Einsum 算子的智能切换
        if is_dynamic:
            # L 形状 [B, T, D, D], Z 形状 [Samples, B, T, D]
            z_correlated = torch.einsum('btij, sbtj -> sbti', L_normalized, z_standard)
        else:
            # L 形状 [B, D, D], Z 形状 [Samples, B, T, D]
            # 静态 Copula 自动把单一协方差矩阵广播给所有 T 时间步
            z_correlated = torch.einsum('bij, sbtj -> sbti', L_normalized, z_standard)
        
        # (3) 还原回真实分布空间: Y = mu + sigma * Z_corr
        y_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * z_correlated
        
        # ---------------------------------------------------------
        # 4. 计算四大指标
        # ---------------------------------------------------------
        mse, mae = calculate_point_metrics(batch_y, mu)
        crps = gaussian_analytical_crps(batch_y, mu, sigma)
        es = energy_score(batch_y, y_samples)
        
        total_mse += mse; total_mae += mae
        total_crps += crps; total_es += es
        num_batches += 1
        
        progress_bar.set_postfix({'MSE': f"{mse:.3f}", 'ES': f"{es:.3f}"})
        
    # ==========================================
    # 5. 汇总与生成可视化报告
    # ==========================================
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches
    avg_crps = total_crps / num_batches
    avg_es = total_es / num_batches
    
    print("\n" + "="*50)
    print("测试集终极评估报告")
    print("="*50)
    print(f"[点预测] MSE:   {avg_mse:.4f}")
    print(f"[点预测] MAE:   {avg_mae:.4f}")
    print(f"[边缘概率] CRPS: {avg_crps:.4f}")
    print(f"[联合概率] ES:   {avg_es:.4f}")
    print("="*50)
    
    # 计算测试集的经验相关系数矩阵 (Empirical Correlation Matrix)
    y_true_concat = np.concatenate(all_y_true_flat, axis=0) # [Total_Samples, D]
    R_empirical = np.corrcoef(y_true_concat.T) # 注意 numpy 的 corrcoef 期望输入是 [Variables, Observations]
    
    # 计算模型预测的平均相关系数矩阵
    R_pred_mean = np.mean(all_R_pred, axis=0)
    
    # 出图！
    plot_correlation_matrices(R_empirical, R_pred_mean)
        
    return avg_mse, avg_crps, avg_es

def plot_correlation_matrices(R_empirical, R_pred):
    """
    绘制 经验相关性 vs 模型预测相关性 热力图 (自动适配超高维防爆版)
    """
    D = R_empirical.shape[0]
    
    # 1. 安全计算画布大小，设置强制上限防爆
    # 单图最大限制在 20 英寸，缩小动态比例
    max_fig_size = 20
    fig_size = min(max_fig_size, max(10, D * 0.12)) 
    
    fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))
    
    # 2. 高维渲染智能降级策略
    use_annot = (D <= 15)       # 超过 15 维关闭数字标注
    grid_lw = 0.5 if D <= 30 else 0  # 超过 30 维必须关闭网格线，防止颜色被边框线糊死
    
    # 3. 绘制经验相关性 (Ground Truth)
    sns.heatmap(R_empirical, annot=use_annot, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, ax=axes[0], square=True, 
                linewidths=grid_lw, cbar_kws={"shrink": .8})
    axes[0].set_title(f"Empirical Correlation (D={D})", fontsize=18, pad=15)
    
    # 4. 绘制大模型预测的相关性
    sns.heatmap(R_pred, annot=use_annot, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, ax=axes[1], square=True, 
                linewidths=grid_lw, cbar_kws={"shrink": .8})
    axes[1].set_title(f"LLM-Copula Predicted Correlation (D={D})", fontsize=18, pad=15)
    
    plt.tight_layout()
    
    # 保存图片 (保留你原本的特定路径)
    save_path = f"./outputs/results/real_data_correlation_{D}D.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📸 {D}维相关系数对比热力图已成功保存为 '{save_path}'！")   
    #plt.show()
