"""
TAID (Temperature-Aware Interpolation Distillation) trainer implementation.
使用分段线性插值（piecewise linear interpolation）实现 TAID。
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import itertools
import gc

from ..models.lora import build_lora_student_model, build_teacher_model
from ..utils.loss import compute_loss_with_taid, compute_taid_lambda


class TAIDDistillationTrainer:
    """TAID 知识蒸馏训练器"""
    
    def __init__(self, config, processor, train_dl, val_dl, device):
        self.cfg = config
        self.processor = processor
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        
        # 训练历史记录
        self.history_steps = []
        self.history_train = []
        self.history_val = []
        self.history_lambda = []  # 记录 TAID lambda 值
        
        # 构建教师模型（冻结）
        self.teacher = build_teacher_model(config, device)
        
        # 学生模型和投影层将在训练时构建
        self.student = None
        self.proj = None
        
        # 梯度缩放器
        self.scaler = GradScaler()
    
    def train(self):
        """训练循环"""
        adapter_dir = os.path.join(self.cfg.output.dir, "adapter")
        ckpt_path = os.path.join(self.cfg.output.dir, "checkpoint.pt")
        step, state = 0, None

        # 检查是否有检查点
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            print("Resuming from checkpoint...")
            self.student, self.proj = build_lora_student_model(self.cfg, self.device)
            self.student.load_state_dict(state["model"])
            if self.proj and "projection" in state:
                self.proj.load_state_dict(state["projection"])
            step = state.get("step", 0)
        else:
            self.student, self.proj = build_lora_student_model(self.cfg, self.device)
            
        # 打印可训练参数数量
        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        if self.proj:
            trainable += sum(p.numel() for p in self.proj.parameters())
        total = sum(p.numel() for p in self.student.parameters()) + \
                (sum(p.numel() for p in self.proj.parameters()) if self.proj else 0)
        print(f"Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")

        # 优化器和调度器
        params = list(self.student.parameters()) + ([] if not self.proj else list(self.proj.parameters()))
        opt = torch.optim.AdamW(params, lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.weight_decay)
        sch = get_linear_schedule_with_warmup(
            opt, 
            self.cfg.training.warmup_steps, 
            self.cfg.training.max_steps
        )
        
        if state:
            opt.load_state_dict(state["optimizer"])
            sch.load_state_dict(state["scheduler"])

        # 训练循环
        pbar = tqdm(total=self.cfg.training.max_steps, initial=step, desc="Training TAID LoRA")
        for step in range(step, self.cfg.training.max_steps):
            self.student.train()
            if self.proj:
                self.proj.train()

            batch = next(itertools.cycle(self.train_dl))
            feats = batch["input_features"].half().to(self.device)
            labels = batch["labels"].to(self.device)
            mask = (feats.sum(1) != 0).long()

            # 前向传播和损失计算
            with autocast():
                loss = compute_loss_with_taid(
                    self.student, self.teacher, self.proj,
                    feats, labels, mask, step,
                    self.cfg, self.device
                ) / self.cfg.training.grad_accum

            # 记录训练损失和 lambda 值
            if (step + 1) % self.cfg.training.log_steps == 0:
                self.history_steps.append(step + 1)
                self.history_train.append(loss.item() * self.cfg.training.grad_accum)
                lambda_val = compute_taid_lambda(step, self.cfg)
                self.history_lambda.append(lambda_val)

            # 反向传播
            self.scaler.scale(loss).backward()
            if (step + 1) % self.cfg.training.grad_accum == 0:
                self.scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, self.cfg.training.max_grad_norm)
                self.scaler.step(opt)
                self.scaler.update()
                sch.step()
                opt.zero_grad(set_to_none=True)

            # 验证评估
            if (step + 1) % self.cfg.training.eval_steps == 0:
                self.student.eval()
                if self.proj:
                    self.proj.eval()
                vals = []
                with torch.no_grad():
                    for vb in self.val_dl:
                        vf = vb['input_features'].half().to(self.device)
                        vl = vb['labels'].to(self.device)
                        m_ = (vf.sum(1) != 0).long()
                        vals.append(compute_loss_with_taid(
                            self.student, self.teacher, self.proj,
                            vf, vl, m_, step,
                            self.cfg, self.device
                        ).item())
                vloss = float(np.mean(vals))
                self.history_val.append(vloss)
                lambda_val = compute_taid_lambda(step, self.cfg)
                print(f"[Step {step+1}] Val Loss: {vloss:.4f}, TAID λ: {lambda_val:.3f}")

            # 打印训练进度
            if (step + 1) % 50 == 0:
                lambda_val = compute_taid_lambda(step, self.cfg)
                print(f"[Step {step+1}/{self.cfg.training.max_steps}] "
                      f"Loss: {loss.item()*self.cfg.training.grad_accum:.4f}, "
                      f"TAID λ: {lambda_val:.3f}")
            pbar.update(1)
        pbar.close()

        # 保存模型
        print("Training completed. Saving adapter...")
        os.makedirs(adapter_dir, exist_ok=True)
        self.student.save_pretrained(adapter_dir)
        self.processor.save_pretrained(adapter_dir)
        if self.proj:
            torch.save(self.proj.state_dict(), os.path.join(adapter_dir, "projection.pt"))

        # 保存训练历史（包含 TAID lambda 值）
        with open(os.path.join(self.cfg.output.dir, "training_history.json"), "w") as hf:
            history = []
            for i in range(len(self.history_steps)):
                entry = {
                    "step": self.history_steps[i],
                    "train_loss": round(self.history_train[i], 4),
                    "taid_lambda": round(self.history_lambda[i], 4)
                }
                if i < len(self.history_val):
                    entry["val_loss"] = round(self.history_val[i], 4)
                history.append(entry)
            
            json.dump(
                {"training_history": history},
                hf,
                indent=2
            )

        print("Done.")
    
    def cleanup(self):
        """清理资源"""
        del self.student
        del self.teacher
        if self.proj:
            del self.proj
        torch.cuda.empty_cache()
        gc.collect() 