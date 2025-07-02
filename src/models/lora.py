"""
LoRA (Low-Rank Adaptation) implementation.
""" 

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Optional
import itertools


def build_lora_student_model(
    config,
    device: torch.device
) -> tuple[PeftModel, Optional[nn.Linear]]:
    """构建 LoRA 学生模型"""
    base = WhisperForConditionalGeneration.from_pretrained(
        config.model.student_model, 
        torch_dtype=torch.float16, 
        use_cache=False
    )
    base.gradient_checkpointing_enable()
    
    for p in itertools.chain(base.model.encoder.parameters(), 
                             base.model.decoder.parameters()):
        p.requires_grad = False

    # LoRA 配置
    lcfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none"
    )
    
    student = get_peft_model(base, lcfg).to(device)
    
    proj = nn.Linear(
        config.model.student_hidden_dim, 
        config.model.teacher_hidden_dim
    ).to(device) if config.distillation.hidden_beta > 0 else None
    
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    if proj:
        trainable += sum(p.numel() for p in proj.parameters())
    total = sum(p.numel() for p in student.parameters()) + \
            (sum(p.numel() for p in proj.parameters()) if proj else 0)
    print(f"Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")
    
    return student, proj


def build_teacher_model(
    config,
    device: torch.device
) -> WhisperForConditionalGeneration:
    """构建教师模型"""
    teacher = WhisperForConditionalGeneration.from_pretrained(
        config.model.teacher_model, 
        torch_dtype=torch.float16, 
        use_cache=False
    ).eval()
    
    # 冻结所有参数
    for p in teacher.parameters():
        p.requires_grad = False
    
    return teacher