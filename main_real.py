import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.FMoECmodel import CopulaLLMForecaster
from data.data_provider import get_real_data_loaders
from trainers.trainer import train_copula_llm
from test_pipline_real import evaluate_copula_model
#from theobound import calculate_theoretical_bounds,calculate_es_lower_bound

device = "cuda" if torch.cuda.is_available() else "cpu"
#/home/gaostudent/LeiJia/Sen/FineTuning/DeepSeek-R1-Distill-Qwen-1.5B
#Qwen/Qwen2.5-0.5B
model_id = "Qwen/Qwen2.5-0.5B"  

def seed_everything(seed=42):
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

def save_copula_model(model, save_path="copula_llm_weights.pth"):
    """
    工业级大模型保存方案：仅剥离并保存参与了梯度更新的参数。
    """
    print(f"🔄 正在提取可训练参数...")
    
    # 获取模型完整的 state_dict
    state_dict = model.state_dict()
    
    # 筛选出 requires_grad 为 True 的参数名称
    trainable_param_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    
    # 仅构建包含这些轻量级参数的字典
    save_dict = {
        name: state_dict[name] for name in trainable_param_names
    }
    
    # 保存到本地
    torch.save(save_dict, save_path)
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ 模型已成功保存至: {save_path}")
    print(f"📦 权重文件大小: {file_size_mb:.2f} MB (仅包含 MoE-LoRA 与 Copula 输出头)")


def load_copula_model(model, load_path="copula_llm_weights.pth", device="cuda"):
    """
    加载微调后的权重，安全注入回包含冻结 LLM 的架构中。
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"找不到权重文件：{load_path}")
        
    print(f"⏳ 正在从 {load_path} 加载微调权重...")
    
    # 加载我们提取的轻量级字典 (使用 weights_only=True 防止安全反序列化警告)
    save_dict = torch.load(load_path, map_location=device, weights_only=True)
    
    # 【核心细节】：strict=False 极其重要！
    # 因为 save_dict 里没有大模型的 base 权重，如果不加 strict=False 会直接报错。
    missing_keys, unexpected_keys = model.load_state_dict(save_dict, strict=False)
    
    # 简单的安全性校验：确保意外的多余 key 为空
    if len(unexpected_keys) > 0:
        print(f"⚠️ 警告：发现了不匹配的意外权重 Keys: {unexpected_keys}")
    else:
        print(f"✅ 权重完美加载并注入完毕！可以开始评估。")
        
    return model
# ==========================================
# 4. 主函数：端到端run
# ==========================================
if __name__ == "__main__":
    # 1. 超参数设置
    #D_VAR = 8           # 变量维度，先设小一点方便本地调试
    SEQ_LEN = 36      # 历史窗口长度 ILL:36
    PRED_LEN = 24       # 预测未来长度 96, 192, 336, 720；24, 36, 48, 60
    BATCH_SIZE = 32  
    #TOTAL_STEPS = 5000
    
    # 替换为你本地下载的数据集路径
    #/home/gaostudent/LeiJia/NLP/myproject/proj1_tsLLM/data/all_six_datasets/electricity/electricity.csv
    CSV_PATH = "./data/all_six_datasets/illness/national_illness.csv" 
    
    print("1. 构建真实数据管道...")
    train_loader, val_loader, test_loader, scaler, D_VAR = get_real_data_loaders(
        csv_path=CSV_PATH,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        batch_size=BATCH_SIZE
    )
    
    print("2. 实例化高维 Copula LLM...")
    model = CopulaLLMForecaster(
        #model_name_or_path="Qwen/Qwen2.5-0.5B", # 本地跑通逻辑可以用极小模型
        model_name_or_path=model_id,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        d_var=D_VAR
    )
    
    # 6. 开始训练
    print("3. 启动 Training Loop...")
    trained_model = train_copula_llm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        lr=5e-4, #1e-4
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # ==========================================
    # 7. 保存炼丹成果 (新增)
    # ==========================================
    save_copula_model(trained_model, save_path=
                      "./outputs/save_models/copula_llm_best_epoch_real.pth")
    
    # ==========================================
    # 8. 加载权重并进入测试评估 (可选，展示完整闭环)
    # ==========================================
    # 假设你新开了一个进程，或者要测试刚才存的权重是否有效
    model_for_test = CopulaLLMForecaster(model_name_or_path=model_id, # 本地跑通逻辑可以用极小模型
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        d_var=D_VAR) # 重新实例化
    model_for_test = load_copula_model(model=model_for_test, load_path="./outputs/save_models/copula_llm_best_epoch_real.pth",device=device)
    evaluate_copula_model(model_for_test, test_loader)
    
