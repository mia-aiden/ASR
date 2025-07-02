#!/usr/bin/env python3
"""
Cell 2.5 重构后的使用示例

展示如何使用新的模块化代码，完全保持原始逻辑不变
"""

# ====================================================================
# 这个文件展示了如何在 Jupyter Notebook 中使用重构后的代码
# 替代原始的 Cell 2.5
# ====================================================================

import os
import warnings
import torch
from torch.utils.data import DataLoader

# 设置环境
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================================================================
# 方式 1: 使用 Optuna 进行超参数搜索（可选）
# ====================================================================

def run_hyperparameter_search_example():
    """运行超参数搜索的示例"""
    from scripts.hyperparameter_search import run_hyperparameter_search
    
    # 需要从 Cell 1 获取这些对象
    # train_ds, safe_collate_fn = ... (从 Cell 1 获取)
    
    print("开始超参数搜索...")
    # best_cfg = run_hyperparameter_search(train_ds, safe_collate_fn)
    print("超参数搜索完成")


# ====================================================================
# 方式 2: 直接使用配置文件进行训练（推荐）
# ====================================================================

def run_training_example():
    """使用配置文件进行训练的示例"""
    from src.utils.config import load_config
    from scripts.train import run_training
    
    # 1. 加载配置
    config = load_config()
    
    # 2. 需要从 Cell 1 获取这些对象
    # processor = ... (从 Cell 1 获取)
    # train_ds, val_ds = ... (从 Cell 1 获取) 
    # safe_collate_fn = ... (从 Cell 1 获取)
    
    print("配置加载完成:")
    print(f"- 教师模型: {config.model.teacher_model}")
    print(f"- 学生模型: {config.model.student_model}")
    print(f"- LoRA rank: {config.lora.r}")
    print(f"- 训练步数: {config.training.max_steps}")
    print(f"- 输出目录: {config.output.dir}")
    
    # 3. 开始训练
    print("开始训练...")
    # run_training(config, processor, train_ds, val_ds, safe_collate_fn, device)
    print("训练完成")


# ====================================================================
# 方式 3: 直接使用训练器类（高级用法）
# ====================================================================

def run_trainer_directly_example():
    """直接使用训练器类的示例"""
    from src.utils.config import load_config
    from src.trainers.lora_trainer import LoRADistillationTrainer
    
    # 1. 加载配置
    config = load_config()
    
    # 2. 需要从 Cell 1 获取这些对象
    # processor = ... (从 Cell 1 获取)
    # train_ds, val_ds = ... (从 Cell 1 获取)
    # safe_collate_fn = ... (从 Cell 1 获取)
    
    # 3. 创建数据加载器
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=config.training.batch_size,
    #     shuffle=True,
    #     num_workers=config.training.num_workers,
    #     pin_memory=True,
    #     collate_fn=safe_collate_fn,
    #     persistent_workers=True,
    #     prefetch_factor=2,
    #     drop_last=True
    # )
    
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=config.training.batch_size,
    #     shuffle=False,
    #     num_workers=config.training.num_workers,
    #     pin_memory=True,
    #     collate_fn=safe_collate_fn,
    #     persistent_workers=True,
    #     prefetch_factor=2
    # )
    
    # 4. 创建并运行训练器
    # trainer = LoRADistillationTrainer(
    #     config=config,
    #     processor=processor,
    #     train_dl=train_loader,
    #     val_dl=val_loader,
    #     device=device
    # )
    
    # trainer.train()
    # trainer.cleanup()
    
    print("训练器使用示例")


# ====================================================================
# 在 Jupyter Notebook 中的实际使用方法
# ====================================================================

"""
在 Jupyter Notebook 的 Cell 2.5 中，用以下代码替代原始代码：

# Cell 2.5 - LoRA Knowledge Distillation (重构版本)

import warnings
import torch
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 方式1: 先进行超参数搜索
from scripts.hyperparameter_search import run_hyperparameter_search
best_cfg = run_hyperparameter_search(train_ds, safe_collate_fn)

# 方式2: 或者直接使用配置文件训练
from scripts.train import run_training
from src.utils.config import load_config

config = load_config()
run_training(config, processor, train_ds, val_ds, safe_collate_fn, device)

print("Cell 2.5 completed successfully!")
"""

if __name__ == "__main__":
    print("Cell 2.5 重构后的使用示例")
    print("请查看文件末尾的注释，了解在 Jupyter Notebook 中的实际使用方法")
    
    # 运行示例（需要注释掉实际的训练调用）
    print("\n=== 配置示例 ===")
    run_training_example()
    
    print("\n=== 完成 ===")
    print("代码重构完成，保持了原始 Cell 2.5 的所有逻辑")

"""
Cell 2.5 重构示例 - 使用模块化的组件
"""

# 导入必要的模块
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders_from_config
from scripts.hyperparameter_search import run_hyperparameter_search
from src.trainers.lora_trainer import LoRADistillationTrainer

def main():
    # 假设已经有了数据集 (从 Cell 1 获取)
    # train_ds, val_ds = ...
    
    print("=== Cell 2.5 重构版本 ===")
    
    # 1. 加载基础配置
    config = load_config()
    print(f"使用配置: {config.model.teacher_model} -> {config.model.student_model}")
    
    # 2. 创建 DataLoader (如果需要超参数搜索)
    # 注意：这里需要先有 train_ds, val_ds 和 safe_collate_fn
    
    # 示例：假设有数据集
    # processor = WhisperProcessor.from_pretrained(config.model.student_model)
    # asr_dataloader = ASRDataLoader(processor, batch_size=config.training.batch_size)
    # safe_collate_fn = asr_dataloader.safe_collate_fn
    
    # 3. (可选) 运行超参数搜索
    # print("运行超参数搜索...")
    # optimized_config = run_hyperparameter_search(train_ds, safe_collate_fn)
    # print("超参数搜索完成，配置已更新")
    
    # 4. 重新加载配置 (现在包含最佳参数)
    config = load_config()
    
    # 5. 创建最终的 DataLoader
    # from transformers import WhisperProcessor
    # from src.data.dataloader import ASRDataLoader
    # 
    # processor = WhisperProcessor.from_pretrained(config.model.student_model)
    # asr_dataloader = ASRDataLoader(
    #     processor=processor,
    #     batch_size=config.training.batch_size,
    #     num_workers=config.training.num_workers,
    #     max_frames=config.data.max_frames,
    #     sample_rate=config.data.sample_rate,
    #     pin_memory=config.data.pin_memory,
    #     persistent_workers=config.data.persistent_workers,
    #     prefetch_factor=config.data.prefetch_factor
    # )
    # 
    # train_loader = asr_dataloader.get_loader(train_ds, shuffle=True, drop_last=True)
    # val_loader = asr_dataloader.get_loader(val_ds, shuffle=False, drop_last=False)
    
    # 6. 创建训练器
    # trainer = LoRADistillationTrainer(
    #     config=config,
    #     train_loader=train_loader,
    #     val_loader=val_loader
    # )
    
    # 7. 开始训练
    # trainer.train()
    
    print("模块化重构完成！")
    print("主要组件:")
    print("- configs/default_config.yaml: 统一配置管理")
    print("- src/data/dataloader.py: DataLoader 创建")
    print("- scripts/hyperparameter_search.py: 超参数搜索")
    print("- src/trainers/lora_trainer.py: 训练器")

if __name__ == "__main__":
    main() 