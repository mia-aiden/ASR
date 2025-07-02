import os
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from transformers import get_linear_schedule_with_warmup
from peft import PeftModel
import itertools
import gc

from ..models.lora import build_lora_student_model, build_teacher_model
from ..utils.loss import compute_loss


class LoRADistillationTrainer:
    """LoRA 知识蒸馏训练器
    
    直接使用原生模块，不重复包装函数
    """
    
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
        
        # 构建教师模型（冻结）
        self.teacher = build_teacher_model(config, device)
        
        # 学生模型和投影层将在 build_student 中创建
        self.student = None
        self.proj = None
    

    
    def train(self):
        """训练循环 - 直接使用原生模型调用"""
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

        # 优化器和调度器
        params = list(self.student.parameters()) + ([] if not self.proj else list(self.proj.parameters()))
        opt = torch.optim.AdamW(params, lr=self.cfg.optimizer.lr, weight_decay=self.cfg.optimizer.weight_decay)
        sch = get_linear_schedule_with_warmup(opt, self.cfg.training.warmup_steps, self.cfg.training.max_steps)
        
        if state:
            opt.load_state_dict(state["optimizer"])
            sch.load_state_dict(state["scheduler"])

        # 训练循环
        pbar = tqdm(total=self.cfg.training.max_steps, initial=step, desc="Training LoRA")
        for step in range(step, self.cfg.training.max_steps):
            self.student.train()
            if self.proj:
                self.proj.train()

            batch = next(itertools.cycle(self.train_dl))
            feats = batch["input_features"].half().to(self.device)
            labels = batch["labels"].to(self.device)
            mask = (feats.sum(1) != 0).long()

            # 前向传播 - 直接调用 loss 函数
            with autocast():
                loss = compute_loss(self.student, feats, labels, mask) / self.cfg.training.grad_accum

            # 记录训练损失
            if (step + 1) % self.cfg.training.log_steps == 0:
                self.history_steps.append(step + 1)
                self.history_train.append(loss.item() * self.cfg.training.grad_accum)

            # 反向传播
            loss.backward()
            if (step + 1) % self.cfg.training.grad_accum == 0:
                nn.utils.clip_grad_norm_(params, self.cfg.training.max_grad_norm)
                opt.step()
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
                        
                        # 直接调用 loss 函数进行验证
                        vals.append(compute_loss(self.student, vf, vl, m_).item())
                        
                vloss = float(np.mean(vals))
                self.history_val.append(vloss)
                print(f"[Step {step+1}] Val Loss: {vloss:.4f}")

            # 打印训练进度
            if (step + 1) % 50 == 0:
                print(f"[Step {step+1}/{self.cfg.training.max_steps}] Loss: {loss.item()*self.cfg.training.grad_accum:.4f}")
            
            pbar.update(1)
        
        pbar.close()

        # 保存模型
        print("Training completed. Saving adapter...")
        os.makedirs(adapter_dir, exist_ok=True)
        self.student.save_pretrained(adapter_dir)
        self.processor.save_pretrained(adapter_dir)
        
        if self.proj:
            torch.save(self.proj.state_dict(), os.path.join(adapter_dir, "projection.pt"))

        # 保存训练历史
        with open(os.path.join(self.cfg.output.dir, "training_history.json"), "w") as hf:
            json.dump({
                "steps": self.history_steps,
                "train": self.history_train,
                "val": self.history_val
            }, hf)

        print("Done.")
    
    def cleanup(self):
        """清理资源"""
        del self.student
        del self.teacher
        if self.proj:
            del self.proj
        torch.cuda.empty_cache()
        gc.collect() 