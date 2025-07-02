"""
损失函数实现
"""

import torch
import torch.nn.functional as F
from peft import PeftModel


def compute_loss(student, feats, labels, mask):
    """计算基础损失"""
    core = student.model if isinstance(student, PeftModel) else student
    out = core(
        input_features=feats,
        attention_mask=mask,
        labels=labels,
        output_hidden_states=True
    )
    return out.loss


def compute_taid_lambda(step: int, config) -> float:
    """计算 TAID lambda 值（分段线性插值）"""
    if step <= 0:
        return config.distillation.taid.start
    if step >= config.training.max_steps:
        return config.distillation.taid.end
    
    half = config.training.max_steps / 2
    if step <= half:
        return config.distillation.taid.start + \
               (config.distillation.taid.mid - config.distillation.taid.start) * (step / half)
    else:
        return config.distillation.taid.mid + \
               (config.distillation.taid.end - config.distillation.taid.mid) * ((step - half) / half)


def compute_loss_with_taid(student, teacher, proj, feats, labels, mask, step, config, device):
    """使用 TAID 计算损失"""
    core = student.model if isinstance(student, PeftModel) else student
    student_outputs = core(input_features=feats, attention_mask=mask,
                          labels=labels, output_hidden_states=True)
    total_loss = student_outputs.loss

    if config.distillation.kl_weight > 0:
        student_logits = student_outputs.logits
        with torch.no_grad():
            teacher_outputs = teacher(input_features=feats.cpu(), attention_mask=mask.cpu(), labels=labels.cpu())
            teacher_logits = teacher_outputs.logits.to(device)
            
        # 确保长度和词汇表大小一致
        min_len = min(student_logits.size(1), teacher_logits.size(1))
        min_vocab = min(student_logits.size(2), teacher_logits.size(2))
        student_logits = student_logits[:, :min_len, :min_vocab]
        teacher_logits = teacher_logits[:, :min_len, :min_vocab]
        
        # 计算概率分布
        student_probs = F.softmax(student_logits / config.distillation.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / config.distillation.temperature, dim=-1)
        
        # TAID 插值
        lambda_val = compute_taid_lambda(step, config)
        interp = (1 - lambda_val) * student_probs + lambda_val * teacher_probs
        
        # KL 散度损失
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / config.distillation.temperature, dim=-1),
            interp.detach(),
            reduction='batchmean'
        ) * (config.distillation.temperature ** 2)
        total_loss += config.distillation.kl_weight * kl_loss

    # Hidden states 对齐损失
    if config.distillation.hidden_beta > 0 and proj is not None:
        student_hidden = student_outputs.encoder_last_hidden_state
        with torch.no_grad():
            teacher_full = teacher(input_features=feats.cpu(), attention_mask=mask.cpu(), labels=labels.cpu(),
                                   output_hidden_states=True)
            teacher_hidden = teacher_full.encoder_last_hidden_state.to(device)
        projected = proj(student_hidden)
        L = min(projected.size(1), teacher_hidden.size(1))
        total_loss += config.distillation.hidden_beta * F.mse_loss(projected[:, :L, :], teacher_hidden[:, :L, :])

    return total_loss 