import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


from models.FMoECmodel import CopulaLLMForecaster
from losses.JCloss import JointCopulaLoss,JointCopulaLoss_opt

def train_copula_llm(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    epochs: int = 20, 
    lr: float = 1e-4, 
    device: str = "cuda"
):
    model.to(device)
    
    # ---------------------------------------------------------
    # 1. 优化器与损失函数设置
    # ---------------------------------------------------------
    # 仅将被注入的 MoE-LoRA 模块和自建的 Output Heads 传入优化器
    # 绝不能把冻结的 LLM 参数传进去，否则会报错或极其浪费显存
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # 使用 AdamW 优化器，加入适当的权重衰减防止过拟合
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2) #1e-2
    
    # 使用余弦退火学习率调度器 (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 实例化我们手写的联合 Copula 损失
    #criterion = JointCopulaLoss(eps=1e-6)
    criterion = JointCopulaLoss_opt(eps=1e-6)
    
    # ---------------------------------------------------------
    # 2. 混合精度训练 (AMP) 设置 (如果使用的是 bfloat16 就不需要 GradScaler，
    # 但如果是 float16，强烈建议使用它来防止梯度下溢)
    # ---------------------------------------------------------
    scaler = torch.amp.GradScaler('cuda') 
    
    print(f"Start training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_loss_marginal = 0.0
        train_loss_copula = 0.0
        
        # 使用 tqdm 包装 DataLoader 以显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_x, batch_y in progress_bar:
            # batch_x: [Batch, Seq_Len, D_var]
            # batch_y: [Batch, Pred_Len, D_var] (未来真实值)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # 开启混合精度上下文
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # 1. 前向传播
                mu, sigma, L_unnormalized = model(batch_x)
                
                #用于超高维的损失设置：
                #mse_loss = torch.mean((mu - batch_y) ** 2)
                #total_loss = total_loss +100*mse_loss #放在scaler前面

                #均值锚定的混合解耦损失
                mse_loss = torch.mean((mu - batch_y) ** 2)
                mu_detach = mu.detach()  # 锚定均值，不让它参与联合损失的梯度计算

                # 2. 计算损失
                total_loss, marginal_loss, copula_loss = criterion(
                    y_true=batch_y, 
                    mu=mu_detach,  #mu_detach,mu
                    sigma=sigma, 
                    L_unnormalized=L_unnormalized
                )
                total_loss_mix = total_loss + mse_loss
                


            # 3. 反向传播 (使用 Scaler 防止数值下溢)
            scaler.scale(total_loss_mix).backward()
            
            # ---------------------------------------------------------
            # 核心工程技巧：梯度裁剪 (Gradient Clipping)
            # ---------------------------------------------------------
            # 解码 Cholesky 矩阵和计算 Copula 密度时容易产生梯度毛刺 (Spikes)
            # 必须在 step 之前将梯度裁剪到最大范数 (例如 1.0)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            #if math.isnan(total_loss.item()):
                #print(f"\n[Warning] 发现 NaN Loss！跳过当前 Batch。")
                #optimizer.zero_grad() # 清空这个有毒的梯度
                #continue # 直接进下一个 Batch
            # 4. 更新权重
            scaler.step(optimizer)
            scaler.update()
            
            # 记录损失 (使用 .item() 取出标量，防止显存泄漏)
            train_loss_total += total_loss_mix.item()
            train_loss_marginal += marginal_loss.item()
            train_loss_copula += copula_loss.item()
            
            # 动态更新进度条显示
            progress_bar.set_postfix({
                'Total': f"{total_loss_mix.item():.4f}",
                'Marg': f"{marginal_loss.item():.4f}",
                'Copula': f"{copula_loss.item():.4f}"
            })
            
        # 计算 Epoch 平均损失
        num_batches = len(train_loader)
        avg_train_loss = train_loss_total / num_batches
        avg_marg_loss = train_loss_marginal / num_batches
        avg_cop_loss = train_loss_copula / num_batches
        
        # ---------------------------------------------------------
        # 3. 验证阶段 (Validation Loop)
        # ---------------------------------------------------------
        model.eval()
        val_loss_total = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    mu, sigma, L_unnormalized = model(batch_x)
                    val_total, _, _ = criterion(batch_y, mu, sigma, L_unnormalized)
                    
                val_loss_total += val_total.item()
                
        avg_val_loss = val_loss_total / len(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        print(f"Epoch {epoch+1} Summary: "
              f"Train Loss: {avg_train_loss:.4f} (Marginal: {avg_marg_loss:.4f}, Copula: {avg_cop_loss:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

    print("Training Complete!")
    return model
