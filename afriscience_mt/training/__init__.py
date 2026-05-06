"""Training module for fine-tuning models."""

from .seq2seq_trainer import (
    Seq2SeqTrainer,
    train_seq2seq,
)
from .lora_trainer import (
    LoRATrainer,
    train_lora,
    get_lora_config,
)
from .multilingual import (
    MultilingualDataset,
    MultilingualCollator,
    create_multilingual_dataset,
    create_n2n_dataset,
    generate_all_language_pairs,
    get_all_language_pairs,
    get_zero_shot_pairs,
    get_african_to_african_pairs,
)
from .callbacks import (
    EvaluationCallback,
    SaveBestModelCallback,
    LoggingCallback,
    MemoryMonitorCallback,
)

__all__ = [
    # Seq2Seq
    "Seq2SeqTrainer",
    "train_seq2seq",
    # LoRA
    "LoRATrainer",
    "train_lora",
    "get_lora_config",
    # Multilingual
    "MultilingualDataset",
    "MultilingualCollator",
    "create_multilingual_dataset",
    "create_n2n_dataset",
    "generate_all_language_pairs",
    "get_all_language_pairs",
    "get_zero_shot_pairs",
    "get_african_to_african_pairs",
    # Callbacks
    "EvaluationCallback",
    "SaveBestModelCallback",
    "LoggingCallback",
    "MemoryMonitorCallback",
]
