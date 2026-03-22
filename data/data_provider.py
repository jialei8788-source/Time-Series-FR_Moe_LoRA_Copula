import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class RealTimeSeriesDataset(Dataset):
    """
    底层 Dataset 类：负责滑动窗口截取
    """
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_samples = len(data) - seq_len - pred_len + 1
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y

def get_real_data_loaders(
    csv_path, 
    seq_len, 
    pred_len, 
    batch_size, 
    train_ratio=0.7, 
    val_ratio=0.1, 
    target_cols=None
):
    """
    工业级数据管道：读取 CSV -> 严格时序切分 -> 避免泄露的归一化 -> 构建 Loader
    """
    print(f"📦 正在加载真实数据集: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. 清洗数据：通常第一列是时间戳 'date'，我们需要剔除它，只保留数值特征
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
        
    # 如果指定了 target_cols，则只取部分维度；否则取全维度
    if target_cols is not None:
        df = df[target_cols]
        
    data = df.values
    total_len = len(data)
    d_var = data.shape[1]
    
    # 2. 严格按时间先后顺序划分数据集 (严禁随机打乱！)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"📊 数据维度 D_VAR: {d_var}")
    print(f"   Train 样本数: {len(train_data)}")
    print(f"   Val   样本数: {len(val_data)}")
    print(f"   Test  样本数: {len(test_data)}")
    
    # 3. 核心防线：基于 Train 数据拟合 Scaler，并应用于全局
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    # 4. 构建 Dataset
    train_dataset = RealTimeSeriesDataset(train_scaled, seq_len, pred_len)
    val_dataset = RealTimeSeriesDataset(val_scaled, seq_len, pred_len)
    test_dataset = RealTimeSeriesDataset(test_scaled, seq_len, pred_len)
    
    # 5. 构建 DataLoader (只有 Train 需要 shuffle)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, scaler, d_var