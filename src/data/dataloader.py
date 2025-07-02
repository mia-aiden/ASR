"""
DataLoader implementation for ASR training.
""" 

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

class ASRDataLoader:
    def __init__(
        self,
        processor: WhisperProcessor,
        batch_size: int = 32,
        num_workers: int = 2,
        max_frames: int = 3000,
        sample_rate: int = 16000,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2
    ):
        """初始化时传入所有 DataLoader 需要的参数"""
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

    def collate_fn(self, batch):
        batch = [b for b in batch if b and b["transcription"].strip()]
        if not batch:
            return None

        # 关键修复：逐个样本处理，模拟成功的直接方法
        processed_features = []
        processed_labels = []

        for item in batch:
            audio_array = item["audio"]["array"]
            text = item["transcription"]

            # 用成功的方法单独处理每个样本
            audio_inputs = self.processor.feature_extractor(
                audio_array,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )

            text_inputs = self.processor.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            # 处理特征长度
            feats = audio_inputs.input_features[0]  # 移除batch维度
            if feats.shape[-1] > self.max_frames:
                feats = feats[..., :self.max_frames]
            elif feats.shape[-1] < self.max_frames:
                pad_size = self.max_frames - feats.shape[-1]
                feats = torch.nn.functional.pad(feats, (0, pad_size))

            processed_features.append(feats)

            # 处理labels
            labels = text_inputs.input_ids[0]  # 移除batch维度
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            processed_labels.append(labels)

        # 组合成batch
        batch_features = torch.stack(processed_features, dim=0)
        batch_labels = torch.stack(processed_labels, dim=0)

        return {
            "input_features": batch_features,
            "labels": batch_labels
        }
        
    def safe_collate_fn(self, batch):
        try:
            result = self.collate_fn(batch)
            if result is None:
                # 返回一个有效的空batch
                return {
                    "input_features": torch.zeros(1, 80, self.max_frames),
                    "labels": torch.full((1, 256), -100, dtype=torch.long)
                }
            return result
        except Exception as e:
            print(f"Collate error: {e}")
            return {
                "input_features": torch.zeros(1, 80, self.max_frames),
                "labels": torch.full((1, 256), -100, dtype=torch.long)
            }
        
    def get_loader(self, dataset, shuffle=True, drop_last=True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.safe_collate_fn,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=drop_last
        )