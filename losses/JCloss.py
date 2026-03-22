import torch.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F

class JointCopulaLoss(nn.Module):
    """边缘分布 NLL + 动态 Gaussian Copula NLL"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_true, mu, sigma, L_unnormalized):
        """
        y_true: [Batch, Seq_Len, D] 真实的高维时序数据
        mu, sigma: [Batch, Seq_Len, D] 边缘分布的均值和标准差
        L_unnormalized: [Batch, Seq_Len, D, D] LLM 输出的原始下三角矩阵
        """
        B, T, D = y_true.shape
        # 注意：为了计算图的安全，可以在开头把网络输出统一转为 float32,绝不让大模型的 bfloat16 污染统计函数
        y_true = y_true.float()
        mu = mu.float()
        sigma = sigma.float()
        
        # 1. 计算边缘分布损失
        marginal_dist = dist.Normal(mu, sigma + self.eps)
        marginal_loss = -marginal_dist.log_prob(y_true).sum(dim=-1)
        
        # 2. 准备 Copula 的输入变量 Z
        u = marginal_dist.cdf(y_true)
        
        # ==========================================
        # 【核心防御】：将 clamp 的边界放大到 1e-4！
        # float32 挡不住 1e-6 的舍入误差，会导致 u 变成精准的 1.0
        # ==========================================
        safe_eps = 1e-4 
        u = torch.clamp(u, safe_eps, 1.0 - safe_eps)
        
        standard_normal = dist.Normal(0.0, 1.0)
        z = standard_normal.icdf(u) # 现在 z 绝对不可能算出 infinity 了
        
        # ---------------------------------------------------------
        # 3. 终极免 Cholesky 算子法：直接构建相关性矩阵的 Cholesky 因子
        # ---------------------------------------------------------
        # 强制转为 float32
        L_unnormalized = L_unnormalized.float()
        
        # 严格保证只取下三角部分
        L_tril = torch.tril(L_unnormalized)
        
        # 强迫对角线严格为正 (使用 softplus，加一个极小值防止除零)
        diag_mask = torch.eye(D, device=L_unnormalized.device, dtype=torch.bool).expand(B, T, D, D)
        diag_elements = F.softplus(torch.diagonal(L_tril, dim1=-2, dim2=-1)) + 1e-5 ##高维可以改为1e-2,低维1e-5
        
        # 安全替换对角线
        L_pd = L_tril * (~diag_mask) + torch.diag_embed(diag_elements)
        
        # 【核心降维打击】：计算每一行的 L2 范数，并将每一行归一化
        # 只要 L 的每一行范数为 1，L @ L.T 的对角线就绝对是 1，天然就是相关性矩阵！
        row_norms = torch.norm(L_pd, p=2, dim=-1, keepdim=True)
        
        # 这就是最终的、绝对安全的 Cholesky 因子，无需任何 linalg.cholesky 计算！
        L_normalized = L_pd / (row_norms + 1e-8)

        #归一化可能导致对角线极小。我们强行给对角线加上 1e-2 的偏置。这虽然会让行范数变成 1.0001，但在工程上是完美的岭回归正则化！
        jitter_diag = torch.ones(D, device=L_normalized.device) * 1e-4   #,低维1e-4
        L_normalized = L_normalized + torch.diag_embed(jitter_diag)
        
        # ---------------------------------------------------------
        # 4. 计算 Copula 损失 (直接使用 L_normalized)
        # ---------------------------------------------------------
        # 确保 z 也是 float32
        z = z.float() 
        
        # 直接把 L_normalized 喂给多元高斯，它内部会自动用这个下三角阵，彻底告别正定报错
        mvn_dist = dist.MultivariateNormal(loc=torch.zeros_like(z), scale_tril=L_normalized)
        
        log_prob_mvn = mvn_dist.log_prob(z) # [B, T]
        log_prob_indep = standard_normal.log_prob(z).sum(dim=-1) # [B, T]
        
        copula_loss = -(log_prob_mvn - log_prob_indep) # [B, T]
        
        # 总损失 = 边缘分布损失 + Copula 依赖损失
        #copula_loss = torch.clamp(copula_loss, min=-10.0, max=1000.0)
        #marginal_loss = torch.clamp(marginal_loss, min=-10.0, max=1000.0)
        total_loss = (marginal_loss + copula_loss).mean()
        
        return total_loss, marginal_loss.mean(), copula_loss.mean()


class JointCopulaLoss_opt(nn.Module):
    def __init__(self,eps):
        super(JointCopulaLoss_opt, self).__init__()
        self.eps = eps

    def forward(self, y_true, mu, sigma, L_unnormalized):
        """
        全能加速版 Copula Loss (自动兼容 3D 静态矩阵与 4D 动态矩阵)
        mu, sigma, batch_y: [B, T, D]
        L_unnormalized: [B, D, D] (高维静态) 或 [B, T, D, D] (低维动态)
        """
        B, T, D = y_true.shape
        device = y_true.device
        
        # 探测输入矩阵是否带有时间维度
        is_dynamic = (L_unnormalized.dim() == 4)

        # ==========================================
        # 1. 边缘损失 (Marginal NLL) 
        # ==========================================
        z = (y_true - mu) / (sigma + 1e-8) # [B, T, D]
        marginal_nll = 0.5 * (z ** 2) + torch.log(sigma + 1e-8)
        marginal_loss = torch.sum(marginal_nll, dim=(1, 2)).mean() 

        # ==========================================
        # 2. 构建纯正的相依结构 (兼容 3D 与 4D)
        # ==========================================
        L_tril = torch.tril(L_unnormalized) 
        
        # 智能构建掩码
        if is_dynamic:
            diag_mask = torch.eye(D, device=device, dtype=torch.bool).expand(B, T, D, D)
        else:
            diag_mask = torch.eye(D, device=device, dtype=torch.bool).expand(B, D, D)
            
        diag_elements = F.softplus(torch.diagonal(L_tril, dim1=-2, dim2=-1)) + 1e-5 #1e-2
        L_pd = L_tril * (~diag_mask) + torch.diag_embed(diag_elements)
        
        row_norms = torch.norm(L_pd, p=2, dim=-1, keepdim=True)
        L = L_pd / (row_norms + 1e-8) 

        # ==========================================
        # 3. 极速计算 Copula 损失 (解三角阵黑魔法)
        # ==========================================
        # (A) log|R| 计算
        # torch.diagonal 会提取最后两维
        log_diag_L = torch.log(torch.diagonal(L, dim1=-2, dim2=-1))
        
        if is_dynamic:
            # L 形状 [B, T, D, D], log_det_R 也是时间相关的，形状 [B, T]
            log_det_R_t = 2.0 * torch.sum(log_diag_L, dim=-1) # [B, T]
            sum_log_det_R = torch.sum(log_det_R_t, dim=-1)    # [B]
        else:
            # L 形状 [B, D, D], 时间不变，直接乘 T
            log_det_R_static = 2.0 * torch.sum(log_diag_L, dim=-1) # [B]
            sum_log_det_R = T * log_det_R_static # [B]
        
        # (B) 二次型项 Z^T R^-1 Z
        if is_dynamic:
            # L: [B, T, D, D], 需要把 Z 变成 [B, T, D, 1] 才能批量解方程
            z_reshaped = z.unsqueeze(-1) # [B, T, D, 1]
            x = torch.linalg.solve_triangular(L, z_reshaped, upper=False) # [B, T, D, 1]
            quad_term_R_inv = torch.sum(x.squeeze(-1) ** 2, dim=(1, 2)) # [B]
        else:
            # L: [B, D, D], 让 Z 变成 [B, D, T]，把 T 作为方程的“多右端项”一次性求解！(极限加速)
            z_transposed = z.transpose(1, 2) # [B, D, T]
            x = torch.linalg.solve_triangular(L, z_transposed, upper=False) # [B, D, T]
            quad_term_R_inv = torch.sum(x ** 2, dim=(1, 2)) # [B]
        
        quad_term_I = torch.sum(z ** 2, dim=(1, 2)) # [B]

        # 汇总 Copula NLL
        copula_loss = 0.5 * (sum_log_det_R + quad_term_R_inv - quad_term_I).mean()

        total_loss = (marginal_loss + copula_loss) / (D * T)
        return total_loss, marginal_loss, copula_loss