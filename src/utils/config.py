from dataclasses import dataclass
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    teacher_model: str
    student_model: str
    student_hidden_dim: int
    teacher_hidden_dim: int


@dataclass
class DataConfig:
    root_dir: str
    sample_cap: int
    val_ratio: float
    batch_size: int
    max_frames: int
    sample_rate: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    num_workers: int


@dataclass
class TrainingConfig:
    max_steps: int
    warmup_steps: int
    eval_steps: int
    log_steps: int
    batch_size: int
    grad_accum: int
    max_grad_norm: float


@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float


@dataclass
class LoRAConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]


@dataclass
class DistillationConfig:
    temperature: float
    kl_weight: float
    hidden_beta: float


@dataclass
class OutputConfig:
    dir: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    lora: LoRAConfig
    distillation: DistillationConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从YAML文件加载配置"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在：{yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            optimizer=OptimizerConfig(**config_dict['optimizer']),
            lora=LoRAConfig(**config_dict['lora']),
            distillation=DistillationConfig(**config_dict['distillation']),
            output=OutputConfig(**config_dict['output'])
        )

    def save(self, save_path: str) -> None:
        """保存配置到YAML文件"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'optimizer': self.optimizer.__dict__,
            'lora': self.lora.__dict__,
            'distillation': self.distillation.__dict__,
            'output': self.output.__dict__
        }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def load_config(config_path: str = "configs/default_config.yaml") -> Config:
    """加载配置的便捷函数"""
    return Config.from_yaml(config_path) 