"""
超参数搜索脚本 - 完全保持原始 Cell 2.5 的 Optuna 搜索逻辑不变
"""

import os
import json
import random
import warnings
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, set_seed
from peft import LoraConfig, get_peft_model, TaskType

import optuna
from optuna.samplers import TPESampler

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import load_config


def objective(trial, train_ds, safe_collate_fn, device):
    """Optuna 目标函数 - 使用 YAML 配置"""
    # 加载基础配置
    config = load_config("../configs/default_config.yaml")
    
    # Optuna 搜索的超参数
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.3)
    temperature = trial.suggest_float("temperature", 1.0, 5.0)
    kl_weight = trial.suggest_float("kl_weight", 0.0, 1.0)
    hidden_beta = trial.suggest_float("hidden_beta", 0.0, 3.0)
    grad_accum = trial.suggest_categorical("grad_accum", [1, 2, 4])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    lora_r = trial.suggest_categorical("lora_r", [32, 64, 128])
    lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32])

    # 加载教师和学生模型
    teacher = WhisperForConditionalGeneration.from_pretrained(
        config.model.teacher_model, torch_dtype=torch.float16, use_cache=False
    ).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    base = WhisperForConditionalGeneration.from_pretrained(
        config.model.student_model, torch_dtype=torch.float16, use_cache=False
    )
    for p in itertools.chain(base.model.encoder.parameters(),
                             base.model.decoder.parameters()):
        p.requires_grad = False

    # LoRA 配置
    lcfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=config.lora.target_modules,
        bias="none"
    )
    student = get_peft_model(base, lcfg).to(device)
    proj = nn.Linear(config.model.student_hidden_dim, config.model.teacher_hidden_dim).to(device) \
           if hidden_beta > 0 else None

    opt = torch.optim.AdamW(
        list(student.parameters()) + ([] if proj is None else list(proj.parameters())),
        lr=lr, weight_decay=config.optimizer.weight_decay
    )

    loader = DataLoader(train_ds,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=safe_collate_fn,
                        num_workers=0,
                        drop_last=True)
    
    total_loss, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= 5: 
            break
        feats = batch["input_features"].half().to(device)
        labels = batch["labels"].to(device)
        mask = (feats.sum(1) != 0).long()
        
        with autocast():
            out = student.model(input_features=feats,
                                attention_mask=mask,
                                labels=labels,
                                output_hidden_states=True)
            loss = out.loss
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        total_loss += loss.item()
        count += 1

    # 清理内存
    del teacher, student, base
    if proj:
        del proj
    torch.cuda.empty_cache()

    return total_loss / max(1, count)


def run_hyperparameter_search(train_dataset, collate_function):
    """运行超参数搜索"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置随机种子
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)
    warnings.filterwarnings("ignore")
    
    print(f"Using device: {device}")

    # 创建 Optuna 研究
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(
        lambda trial: objective(trial, train_dataset, collate_function, device),
        n_trials=50
    )

    # 获取基础配置和最佳参数
    config = load_config("../configs/default_config.yaml")
    best_params = study.best_params

    # 更新配置 - 与原始 Notebook 保持一致: cfg = CFG(); for k, v in study.best_params.items(): setattr(cfg, k, v)
    import copy
    updated_config = copy.deepcopy(config)
    
    for k, v in best_params.items():
        if k == 'lr':
            updated_config.optimizer.lr = v
        elif k == 'lora_dropout':
            updated_config.lora.dropout = v
        elif k == 'temperature':
            updated_config.distillation.temperature = v
        elif k == 'kl_weight':
            updated_config.distillation.kl_weight = v
        elif k == 'hidden_beta':
            updated_config.distillation.hidden_beta = v
        elif k == 'grad_accum':
            updated_config.training.grad_accum = v
        elif k == 'batch_size':
            updated_config.training.batch_size = v
        elif k == 'lora_r':
            updated_config.lora.r = v
        elif k == 'lora_alpha':
            updated_config.lora.alpha = v

    # 保存配置
    os.makedirs(config.output.dir, exist_ok=True)
    
    # 保存最佳参数到 JSON（用于记录）
    with open(os.path.join(config.output.dir, "training_config.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    
    # 将最佳参数写回到 default_config.yaml（保持原有格式）
    import yaml
    config_path = "../configs/default_config.yaml"
    
    # 读取原始 YAML 文件
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    # 更新相关参数
    for k, v in best_params.items():
        if k == 'lr':
            yaml_data['optimizer']['lr'] = v
        elif k == 'lora_dropout':
            yaml_data['lora']['dropout'] = v
        elif k == 'temperature':
            yaml_data['distillation']['temperature'] = v
        elif k == 'kl_weight':
            yaml_data['distillation']['kl_weight'] = v
        elif k == 'hidden_beta':
            yaml_data['distillation']['hidden_beta'] = v
        elif k == 'grad_accum':
            yaml_data['training']['grad_accum'] = v
        elif k == 'batch_size':
            yaml_data['training']['batch_size'] = v
        elif k == 'lora_r':
            yaml_data['lora']['r'] = v
        elif k == 'lora_alpha':
            yaml_data['lora']['alpha'] = v
    
    # 写回文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"已将最佳参数更新到 {config_path}")

    # 打印结果
    print("=== Hyperparameters from 50-run auto-search ===")
    print(f"  lr: {best_params['lr']}")
    print(f"  lora_dropout: {best_params['lora_dropout']}")
    print(f"  temperature: {best_params['temperature']}")
    print(f"  kl_weight: {best_params['kl_weight']}")
    print(f"  hidden_beta: {best_params['hidden_beta']}")
    print(f"  grad_accum: {best_params['grad_accum']}")
    print(f"  batch_size: {best_params['batch_size']}")
    print(f"  lora_r: {best_params['lora_r']}")
    print(f"  lora_alpha: {best_params['lora_alpha']}")

    return updated_config