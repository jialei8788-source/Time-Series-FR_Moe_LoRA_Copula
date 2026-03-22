import torch
import torch.distributions as dist
def calculate_point_metrics(y_true, mu):
    """
    点预测指标 (提取概率分布的均值作为预测值)
    用于与 PatchTST, iTransformer 等非概率模型对标
    """
    mse = torch.mean((y_true - mu) ** 2)
    mae = torch.mean(torch.abs(y_true - mu))
    return mse.item(), mae.item()

def energy_score(y_true, samples):
    """
    计算多变量能量分数 (Energy Score)
    y_true: [Batch, Seq_Len, D] 真实值
    samples: [Num_Samples, Batch, Seq_Len, D] 蒙特卡洛采样的预测路径
    """
    num_samples = samples.shape[0]
    
    # 1. 计算样本与真实值之间的 L2 距离期望
    # term1: [Num_Samples, Batch, Seq_Len]
    term1 = torch.norm(samples - y_true.unsqueeze(0), dim=-1) 
    expected_dist_to_true = torch.mean(term1, dim=0) # [Batch, Seq_Len]
    
    # 2. 计算样本自身之间的 L2 距离期望 (反映预测的不确定性/发散度)
    # 为了高效计算，打乱样本顺序进行配对计算近似
    samples_shifted = samples[torch.randperm(num_samples)]
    term2 = torch.norm(samples - samples_shifted, dim=-1)
    expected_dist_between_samples = torch.mean(term2, dim=0) # [Batch, Seq_Len]
    
    # ES = E||Y_hat - Y|| - 0.5 * E||Y_hat - Y_hat'||
    es = expected_dist_to_true - 0.5 * expected_dist_between_samples
    
    return es.mean().item()

def gaussian_analytical_crps(y_true, mu, sigma):
    """
    高斯分布的解析版 CRPS (连续排位概率分数)
    y_true, mu, sigma: [Batch, Seq_Len, D]
    """
    # 将真实值标准化
    z = (y_true - mu) / sigma
    
    # 计算标准正态的 PDF 和 CDF
    standard_normal = dist.Normal(0.0, 1.0)
    pdf_z = torch.exp(standard_normal.log_prob(z))
    cdf_z = standard_normal.cdf(z)
    
    # CRPS 解析公式: sigma * [ z * (2*CDF(z) - 1) + 2*PDF(z) - 1/sqrt(pi) ]
    crps = sigma * (
        z * (2 * cdf_z - 1.0) + 
        2 * pdf_z - 
        1.0 / torch.sqrt(torch.tensor(torch.pi))
    )
    return crps.mean().item()