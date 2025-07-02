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