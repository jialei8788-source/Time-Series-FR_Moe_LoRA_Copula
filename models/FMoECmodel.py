import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, BitsAndBytesConfig
# 假设 FreqRouter, MoELoRALayer, RevIN 已经在上一步定义好了
from models.my_modules import FreqRouter, MoELoRALayer, RevIN

class CopulaLLMForecaster(nn.Module):
    def __init__(self, model_name_or_path, seq_len, pred_len, d_var, num_experts=4):
        super().__init__()
        
        # =========================================================
        # 【架构级自适应开关】：根据维度自动决定是否开启时变 Copula
        # =========================================================
        self.dynamic_copula = True if d_var < 50 else False
        
        self.dropout = nn.Dropout(0.2)
        config = AutoConfig.from_pretrained(model_name_or_path)
        d_model = config.hidden_size 
        print(f"Auto-detected d_model: {d_model} for {model_name_or_path}")
        self.seq_len = seq_len      
        self.pred_len = pred_len    
        self.d_var = d_var          
        
        # 1. 频域处理与路由模块
        self.patch_proj = nn.Linear(d_var, d_model)
        freq_dim = (seq_len // 2 + 1) * d_var
        self.freq_router = FreqRouter(freq_dim=freq_dim, num_experts=num_experts)
        
        # ---------------------------------------------------------
        # 2. 核心大语言模型 (使用 bitsandbytes 开启 4-bit QLoRA 加载)
        # ---------------------------------------------------------
        print(f"Loading base LLM from {model_name_or_path} with 4-bit quantization...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.llm = AutoModel.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        # 冻结基座模型所有参数
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # 注入 MoE-LoRA 模块到 LLM 的每一层 Attention 中
        self._inject_moe_lora(num_experts)
        
        # ---------------------------------------------------------
        # 3. 解耦的联合概率输出头
        # ---------------------------------------------------------
        def build_mlp_head(in_dim, out_dim, hidden_dim=256):
            return nn.Sequential(
                   nn.Linear(in_dim, hidden_dim),
                   nn.SiLU(), 
                   nn.Linear(hidden_dim, out_dim)
                   )
                   
        out_dim = pred_len * d_var
        self.head_mu = build_mlp_head(d_model, out_dim)
        self.head_sigma = build_mlp_head(d_model, out_dim)
        
        l_elements = d_var * (d_var + 1) // 2
        # 根据动态开关，决定最终输出多少个矩阵的元素
        matrix_time_steps = pred_len if self.dynamic_copula else 1
        self.head_L_vec = build_mlp_head(d_model, matrix_time_steps * l_elements) 
        
        # =========================================================
        # 权重初始化安全防线
        # =========================================================
        nn.init.normal_(self.head_mu[-1].weight, std=0.01)
        nn.init.normal_(self.head_sigma[-1].weight, std=0.01)
        
        # 【NaN/OOM 拯救者】：无论是高维还是低维，强制零初始化保单位阵 (R = I)
        nn.init.zeros_(self.head_L_vec[-1].weight) 
        nn.init.zeros_(self.head_L_vec[-1].bias)   

        # 实例化 RevIN
        self.revin = RevIN(num_features=d_var)


    def _inject_moe_lora(self, num_experts):
        """遍历 LLM 层，将原来的 q_proj, v_proj 替换为包含 MoE-LoRA 的包装器"""
        for i, layer in enumerate(self.llm.layers):
            target_q = layer.self_attn.q_proj
            target_v = layer.self_attn.v_proj
            
            layer.self_attn.q_proj = MoELoRAWrapper(target_q, num_experts)
            layer.self_attn.v_proj = MoELoRAWrapper(target_v, num_experts)
            
    def forward(self, x):
        """
        x 形状: [Batch, Seq_Len, D_var]
        """
        B, T, D = x.shape
        
        # =========================================================
        # RevIN 归一化输入
        # =========================================================
        x_norm = self.revin(x, mode='norm')
        
        # ---------------------------------------------------------
        # Step 1: 频域路由权重计算 
        # ---------------------------------------------------------
        x_fft = torch.fft.rfft(x_norm, dim=1).abs() # [B, T//2+1, D]
        x_fft_flat = x_fft.reshape(B, -1)
        
        routing_weights = self.freq_router(x_fft_flat) 
        MoELoRAWrapper.current_routing_weights = routing_weights
        
        # ---------------------------------------------------------
        # Step 2: 时域投影与 LLM 推理 
        # ---------------------------------------------------------
        x_emb = self.patch_proj(x_norm) 
        
        llm_out = self.llm(inputs_embeds=x_emb)
        last_hidden = self.dropout(llm_out.last_hidden_state[:, -1, :])
        
        # ---------------------------------------------------------
        # Step 3: 多元联合概率参数解码与反归一化
        # ---------------------------------------------------------
        # 1. 解码 mu
        mu_norm = self.head_mu(last_hidden).view(B, self.pred_len, D)
        mu = self.revin(mu_norm, mode='denorm')
        
        # 2. 解码 sigma
        raw_sigma = self.head_sigma(last_hidden).view(B, self.pred_len, D)
        
        time_penalty = torch.linspace(0.05, 0.2, steps=self.pred_len, device=last_hidden.device)
        time_penalty = time_penalty.view(1, self.pred_len, 1) 
        
        sigma_norm = F.softplus(raw_sigma) + time_penalty
        sigma = sigma_norm * self.revin.stdev
        
        # =========================================================
        # 【架构分流】：解码动态相关性矩阵
        # 核心逻辑：高维保命挤压时间，低维全域动态展开
        # =========================================================
        if self.dynamic_copula:
            # 【低维分支】：提取随时间演变的矩阵序列 [B, pred_len, -1]
            L_vec = self.head_L_vec(last_hidden).view(B, self.pred_len, -1)
            # 重构为 4D 动态下三角阵 [B, pred_len, D, D]
            L_unnormalized = self._vector_to_tril(L_vec, B, self.pred_len, D)
        else:
            # 【高维分支】：提取唯一全局矩阵 [B, 1, -1]
            L_vec = self.head_L_vec(last_hidden).view(B, 1, -1)
            # 重构为 3D 静态下三角阵 [B, D, D]，利用 squeeze 彻底干掉时间维度
            L_unnormalized = self._vector_to_tril(L_vec, B, 1, D).squeeze(1) 
        
        return mu, sigma, L_unnormalized

    def _vector_to_tril(self, vec, B, H, D):
        """辅助函数：将预测的扁平向量转换为下三角矩阵格式"""
        L = torch.zeros(B, H, D, D, device=vec.device, dtype=vec.dtype)
        tril_indices = torch.tril_indices(row=D, col=D, offset=0)
        L[:, :, tril_indices[0], tril_indices[1]] = vec
        return L

# ---------------------------------------------------------
# 包装器类 (用于替换 HF 底层 Linear)
# ---------------------------------------------------------
class MoELoRAWrapper(nn.Module):
    current_routing_weights = None 

    def __init__(self, base_linear, num_experts):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False
            
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        
        self.moe_lora = MoELoRALayer(in_features, out_features, num_experts)
        
    def forward(self, x):
        base_out = self.base_linear(x)
        lora_out = self.moe_lora(x, MoELoRAWrapper.current_routing_weights)
        return base_out + lora_out