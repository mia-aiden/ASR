"""
使用更新后的配置和 DataLoader 的示例
"""

from src.utils.config import load_config
from src.data.dataloader import ASRDataLoader
from transformers import WhisperProcessor

def main():
    print("=== 配置和 DataLoader 使用示例 ===\n")
    
    # 1. 加载配置
    config = load_config()
    print("✅ 配置加载成功!")
    print(f"   数据配置: max_frames={config.data.max_frames}, sample_rate={config.data.sample_rate}")
    print(f"   训练配置: batch_size={config.training.batch_size}, num_workers={config.training.num_workers}")
    print(f"   DataLoader配置: pin_memory={config.data.pin_memory}, persistent_workers={config.data.persistent_workers}")
    print(f"   LoRA配置: r={config.model.lora_r}, alpha={config.model.lora_alpha}, dropout={config.model.lora_dropout}")
    print()
    
    # 2. 创建 processor
    print("✅ 创建 WhisperProcessor...")
    processor = WhisperProcessor.from_pretrained(config.model.student_model)
    print(f"   使用模型: {config.model.student_model}")
    print()
    
    # 3. 创建 ASRDataLoader
    print("✅ 创建 ASRDataLoader...")
    asr_dataloader = ASRDataLoader(
        processor=processor,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        max_frames=config.data.max_frames,
        sample_rate=config.data.sample_rate,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor
    )
    print(f"   DataLoader 参数:")
    print(f"     - batch_size: {asr_dataloader.batch_size}")
    print(f"     - num_workers: {asr_dataloader.num_workers}")
    print(f"     - max_frames: {asr_dataloader.max_frames}")
    print(f"     - sample_rate: {asr_dataloader.sample_rate}")
    print(f"     - pin_memory: {asr_dataloader.pin_memory}")
    print(f"     - persistent_workers: {asr_dataloader.persistent_workers}")
    print(f"     - prefetch_factor: {asr_dataloader.prefetch_factor}")
    print()
    
    # 4. 展示如何创建不同类型的 DataLoader
    print("✅ DataLoader 创建方式:")
    print("   # 假设已有数据集 train_ds, val_ds, test_ds")
    print("   train_loader = asr_dataloader.get_loader(train_ds, shuffle=True, drop_last=True)")
    print("   val_loader = asr_dataloader.get_loader(val_ds, shuffle=False, drop_last=False)")
    print("   test_loader = asr_dataloader.get_loader(test_ds, shuffle=False, drop_last=False)")
    print()
    
    print("✅ 所有组件初始化完成!")
    print("\n对应原始 Cell 2.5 中的:")
    print("   train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, ...)")
    print("   val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, ...)")

if __name__ == "__main__":
    main() 