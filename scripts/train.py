import os
import gc
import torch
from transformers import WhisperProcessor, set_seed
from typing import Optional, Dict, Any

# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/root/.cache/huggingface/transformers'

from src.utils.config import load_config
from src.data.dataset import load_dataset
from src.data.dataloader import ASRDataLoader
from src.trainers.lora_trainer import LoRADistillationTrainer

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
    
    # 1. 加载配置和数据
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # 2. 加载数据集和处理器
    processor = WhisperProcessor.from_pretrained(cfg.model.teacher_model)
    train_ds, val_ds = load_dataset(
        root_dir=cfg.data.root_dir,
        processor=processor,
        sample_cap=cfg.data.sample_cap,
        val_ratio=cfg.data.val_ratio
    )
    
    # 3. 创建数据加载器（用于超参数搜索和训练）
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
    
    # 4. 第一步：超参数优化
    print("\n第一步: 开始超参数优化...")
    print("正在运行 50 次试验以找到最佳超参数...")
    
    # 获取 collate function
    safe_collate_fn = data_loader.safe_collate_fn
    
    # 运行超参数搜索（使用已有的函数）
    best_params = run_hyperparameter_search(train_ds, safe_collate_fn)
    print("✓ 超参数优化完成，最佳参数已更新到配置文件")
    
    # 5. 重新加载更新后的配置
    cfg = load_config(config_path)
    
    # 6. 第二步：使用最佳参数进行训练
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
    
    # 7. 创建训练器并开始训练
    trainer = LoRADistillationTrainer(
        config=cfg,
        processor=processor,
        train_dl=train_loader,
        val_dl=val_loader,
        device=device
    )
    
    # 8. 开始训练
    best_metric = trainer.train()
    
    # 9. 清理资源
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n========================================")
    print("✓ 完整训练流程成功完成！")
    print("  1. 超参数优化 ✓")
    print("  2. 模型训练 ✓")
    print("========================================")
    
    return best_metric

if __name__ == "__main__":
    run_training()

