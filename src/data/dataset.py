"""
Dataset implementation for LibriSpeech ASR.
""" 

import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch
from datasets import Dataset, Audio
from sklearn.model_selection import train_test_split

def load_dataset(
    root_dir: str,
    processor: Any,
    sample_cap: Optional[int] = None,
    val_ratio: float = 0.2
) -> Tuple[Dataset, Dataset]:
    """加载并准备数据集
    
    Args:
        root_dir: 数据集根目录
        processor: Whisper处理器
        sample_cap: 样本数量上限
        val_ratio: 验证集比例
        
    Returns:
        训练集和验证集的元组
    """
    dataset = LibriSpeechDataset(
        root_dir=root_dir,
        sample_cap=sample_cap,
        val_ratio=val_ratio
    )
    train_ds, val_ds, _ = dataset.prepare_datasets()
    return train_ds, val_ds

class LibriSpeechDataset:
    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 16000,
        sample_cap: Optional[int] = None,
        val_ratio: float = 0.2
    ):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.sample_cap = sample_cap
        self.val_ratio = val_ratio
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        
    def load_split(self, splits: list, cap: int = None) -> Dataset:
        audio_paths, transcripts = [], []
        for split in splits:
            split_dir = os.path.join(self.root_dir, split)
            if not os.path.isdir(split_dir):
                print(f"Warning: missing {split_dir}, skipping")
                continue
            for flac_path in glob.glob(f"{split_dir}/**/*.flac", recursive=True):
                stem = Path(flac_path).stem
                # find matching transcript line
                for txt in glob.glob(f"{os.path.dirname(flac_path)}/*.trans.txt"):
                    with open(txt, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith(stem):
                                text = line.strip().split(" ", 1)[1]
                                audio_paths.append(flac_path)
                                transcripts.append(text)
                                break
                    if len(audio_paths) >= (cap or float("inf")):
                        break
                if cap and len(audio_paths) >= cap:
                    break
            if cap and len(audio_paths) >= cap:
                break
        if not audio_paths:
            raise ValueError(f"No audio files found under {self.root_dir} for {splits}")
        print(f"Found {len(audio_paths)} audio files")
        ds = Dataset.from_dict({"audio": audio_paths, "transcription": transcripts})
        return ds.cast_column("audio", Audio(sampling_rate=self.sample_rate))

    def prepare_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """准备训练、验证和测试数据集"""
        # 加载数据
        train_val_ds = self.load_split(["train-clean-100", "dev-clean"], cap=self.sample_cap)
        test_ds = self.load_split(["test-clean"], cap=None)
        
        # 分割训练和验证集
        idx = list(range(len(train_val_ds)))
        train_idx, val_idx = train_test_split(
            idx, 
            test_size=self.val_ratio, 
            random_state=42, 
            shuffle=True
        )
        
        self.train_ds = train_val_ds.select(train_idx)
        self.val_ds = train_val_ds.select(val_idx)
        self.test_ds = test_ds
        
        print(f"Samples → train: {len(self.train_ds)}, "
              f"val: {len(self.val_ds)}, test: {len(self.test_ds)}")
        
        return self.train_ds, self.val_ds, self.test_ds
    
    @staticmethod
    def summarize_real_durations(ds, name, sr=16_000):
        if not isinstance(ds, Dataset):
            raise RuntimeError(f"{name}_ds 不是 HuggingFace Dataset，请重新运行 Cell 1")
        # 1) 把路径字段换成真正的 audio 数组
        ds2 = ds.cast_column("audio", Audio(sampling_rate=sr))
        # 2) 并行批量计算每条的时长，然后立即删除 audio 字段
        ds2 = ds2.map(
            lambda batch: {
                "duration": [
                    len(item["array"]) / sr
                    for item in batch["audio"]
                ]
            },
            batched=True,
            batch_size=32,
            num_proc=4,
            remove_columns=["audio"],
            load_from_cache_file=False,
        )
        # 3) 汇总打印
        durations = ds2["duration"]
        total_h   = sum(durations) / 3600.0
        avg_s     = sum(durations) / len(durations)
        print(f"{name:5s} | Samples: {len(ds2):4d} | Avg: {avg_s:5.1f}s | Total: {total_h:5.1f}h")