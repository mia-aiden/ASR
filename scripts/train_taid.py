"""
使用 TAID (Temperature-Aware Interpolation Distillation) 进行 LoRA 知识蒸馏训练。
包含超参数调优和完整训练流程。
"""

import os
import gc
import json
import torch
from transformers import WhisperProcessor, set_seed

from src.data.dataset import load_dataset
from src.data.dataloader import ASRDataLoader
from src.utils.config import load_config
from src.trainers.taid_trainer import TAIDDistillationTrainer

# 导入超参数搜索功能
from scripts.hyperparameter_search import run_hyperparameter_search

def run_training(config_path: str = "../configs/default_config.yaml") -> float:
    """运行完整训练流程：超参数优化 + 训练
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        float: 验证集上的最佳性能指标
    """
    print("========================================")
    print("开始完整训练流程")
    print("========================================")
    
    # 1. 设置环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # 2. 加载配置
    cfg = load_config(config_path)
    
    # 3. 加载数据集和处理器
    processor = WhisperProcessor.from_pretrained(cfg.model.student_model)
    train_ds, val_ds = load_dataset(
        root_dir=cfg.data.root_dir,
        processor=processor,
        sample_cap=cfg.data.sample_cap,
        val_ratio=cfg.data.val_ratio
    )
    
    # 4. 创建数据加载器
    data_loader = ASRDataLoader(
        processor=processor,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        max_frames=cfg.data.max_frames,
        sample_rate=cfg.data.sample_rate,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor
    )
    
    # 5. 第一步：超参数优化
    print("\n第一步: 开始超参数优化...")
    best_params = run_hyperparameter_search(train_ds, data_loader.safe_collate_fn)
    print("✓ 超参数优化完成，最佳参数已更新到配置文件")
    
    # 6. 重新加载更新后的配置
    cfg = load_config(config_path)
    
    # 7. 第二步：使用最佳参数进行训练
    print("\n第二步: 开始正式训练...")
    print("使用优化后的超参数进行模型训练...")
    
    # 重新创建数据加载器（使用新的batch_size）
    data_loader = ASRDataLoader(
        processor=processor,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        max_frames=cfg.data.max_frames,
        sample_rate=cfg.data.sample_rate,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor
    )
    train_loader = data_loader.get_loader(train_ds, shuffle=True, drop_last=True)
    val_loader = data_loader.get_loader(val_ds, shuffle=False, drop_last=False)
    
    # 8. 创建训练器并开始训练
    trainer = TAIDDistillationTrainer(cfg, processor, train_loader, val_loader, device)
    best_metric = trainer.train()
    
    # 9. 清理资源
    trainer.cleanup()
    torch.cuda.empty_cache()
    gc.collect()
    
    # 10. 显示训练结果
    out_dir = cfg.output.dir
    print("\n训练完成！")
    print("输出目录内容:", os.listdir(out_dir))
    print("适配器目录内容:", os.listdir(os.path.join(out_dir, "adapter")))
    
    # 11. 显示 TAID lambda 进展
    hist = json.load(open(os.path.join(out_dir, "training_history.json")))
    if "taid_lambda" in hist:
        print("\nTAID Lambda 进展:")
        for i in range(0, len(hist["taid_lambda"]), max(1, len(hist["taid_lambda"])//5)):
            step = hist["steps"][i]
            lambda_val = hist["taid_lambda"][i]
            print(f"  Step {step}: λ = {lambda_val:.3f}")
    
    print("\n========================================")
    print("✓ 完整训练流程成功完成！")
    print("  1. 超参数优化 ✓")
    print("  2. 模型训练 ✓")
    print("========================================")
    
    return best_metric

if __name__ == "__main__":
    run_training()
