import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqRouter(nn.Module):
    """基于频域特征的门控路由网络"""
    def __init__(self, freq_dim, num_experts):
        super().__init__()
        # 简单高效的单层路由，输出各个专家的权重
        self.classifier = nn.Linear(freq_dim, num_experts)
        
    def forward(self, freq_features):
        # freq_features: [Batch, freq_dim]
        logits = self.classifier(freq_features)
        # 使用 Softmax 归一化为概率权重
        routing_weights = F.softmax(logits, dim=-1)
        return routing_weights # [Batch, num_experts]

class MoELoRALayer(nn.Module):
    """动态 MoE-LoRA 适配器，用于替换或注入到 LLM 的 Attention 层中"""
    def __init__(self, in_features, out_features, num_experts, rank=8, alpha=16.0):
        super().__init__()
        self.num_experts = num_experts
        self.rank = rank
        self.scaling = alpha / rank
        
        # 将多个专家的 LoRA 权重堆叠在一起，方便向量化计算 (Einsum)
        # lora_A 负责降维，lora_B 负责升维
        self.lora_A = nn.Parameter(torch.randn(num_experts, in_features, rank) / rank)
        self.lora_B = nn.Parameter(torch.zeros(num_experts, rank, out_features))
        
    def forward(self, x, routing_weights):
        """
        x: [Batch, Seq_Len, in_features] 大模型的隐状态输入
        routing_weights: [Batch, num_experts] 频域路由器输出的权重
        """
        B, Seq_Len, _ = x.shape
        
        # 为了高效计算，我们使用 torch.einsum 避免低效的 for 循环遍历专家
        # 1. 计算每个专家的 lora_A 映射: [Batch, Seq_Len, num_experts, rank]
        xA = torch.einsum('bsi, eir -> bser', x, self.lora_A)
        
        # 2. 计算每个专家的 lora_B 映射: [Batch, Seq_Len, num_experts, out_features]
        xAB = torch.einsum('bser, ero -> bseo', xA, self.lora_B)
        
        # 3. 使用路由权重动态聚合专家的输出
        # routing_weights shape: [Batch, num_experts], 扩展到 [Batch, 1, num_experts, 1]
        weights = routing_weights.unsqueeze(1).unsqueeze(-1)
        
        # 聚合后形状: [Batch, Seq_Len, out_features]
        lora_output = (xAB * weights).sum(dim=2) * self.scaling
        
        return lora_output


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        # 可以加上可学习的仿射变换参数（可选，这里为了稳健先设为 False）
        self.affine = False 

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # x shape: [B, Seq_Len, D]
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        return (x - self.mean) / self.stdev

    def _denormalize(self, x):
        return x * self.stdev + self.mean