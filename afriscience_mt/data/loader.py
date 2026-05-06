"""Data loader utilities for different model types."""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import AfriScienceDataset


def seq2seq_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: Any,
    max_source_length: int = 256,
    max_target_length: int = 256,
    padding: str = "max_length",
    truncation: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for seq2seq models (M2M100, NLLB).

    Args:
        batch: List of samples from dataset
        tokenizer: Tokenizer instance
        max_source_length: Maximum source sequence length
        max_target_length: Maximum target sequence length
        padding: Padding strategy
        truncation: Whether to truncate

    Returns:
        Batch dictionary with input_ids, attention_mask, labels
    """
    sources = [item["source"] for item in batch]
    targets = [item["target"] for item in batch]

    # Tokenize sources
    model_inputs = tokenizer(
        sources,
        max_length=max_source_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

    # Replace padding token id with -100 for loss calculation
    labels["input_ids"] = labels["input_ids"].masked_fill(
        labels["input_ids"] == tokenizer.pad_token_id, -100
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def llm_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int = 512,
    instruction_template: str = "Translate the following {source_lang} text to {target_lang}:",
    padding: str = "max_length",
    truncation: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for LLM fine-tuning.

    Args:
        batch: List of samples from dataset
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        instruction_template: Template for instruction
        padding: Padding strategy
        truncation: Whether to truncate

    Returns:
        Batch dictionary for causal LM training
    """
    texts = []

    for item in batch:
        source_lang = item.get("source_lang", "English")
        target_lang = item.get("target_lang", "")
        source = item["source"]
        target = item["target"]

        # Format as instruction-following
        instruction = instruction_template.format(
            source_lang=source_lang,
            target_lang=target_lang,
        )

        # Create full text for causal LM
        if hasattr(tokenizer, "apply_chat_template"):
            # Use chat template if available
            messages = [
                {"role": "user", "content": f"{instruction}\n\n{source}"},
                {"role": "assistant", "content": target},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Simple format
            text = f"{instruction}\n\nSource: {source}\nTranslation: {target}"

        texts.append(text)

    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
    )

    # For causal LM, labels are the same as input_ids
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    # Mask padding tokens
    model_inputs["labels"] = model_inputs["labels"].masked_fill(
        model_inputs["labels"] == tokenizer.pad_token_id, -100
    )

    return model_inputs


class Seq2SeqCollator:
    """Collator class for seq2seq models."""

    def __init__(
        self,
        tokenizer: Any,
        max_source_length: int = 256,
        max_target_length: int = 256,
        padding: str = "max_length",
        label_pad_token_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return seq2seq_collate_fn(
            batch=batch,
            tokenizer=self.tokenizer,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            padding=self.padding,
        )


class LLMCollator:
    """Collator class for LLM fine-tuning."""

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        instruction_template: str = "Translate the following {source_lang} text to {target_lang}:",
        padding: str = "max_length",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template
        self.padding = padding

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return llm_collate_fn(
            batch=batch,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            instruction_template=self.instruction_template,
            padding=self.padding,
        )


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        collate_fn: Collate function
        pin_memory: Pin memory for GPU
        drop_last: Drop last incomplete batch

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_seq2seq_dataloader(
    dataset: AfriScienceDataset,
    tokenizer: Any,
    batch_size: int = 16,
    max_source_length: int = 256,
    max_target_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create DataLoader for seq2seq models.

    Args:
        dataset: AfriScience dataset
        tokenizer: Seq2seq tokenizer
        batch_size: Batch size
        max_source_length: Maximum source length
        max_target_length: Maximum target length
        shuffle: Whether to shuffle
        num_workers: Number of workers

    Returns:
        DataLoader for seq2seq training
    """
    collator = Seq2SeqCollator(
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )


def create_llm_dataloader(
    dataset: AfriScienceDataset,
    tokenizer: Any,
    batch_size: int = 4,
    max_length: int = 512,
    instruction_template: Optional[str] = None,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create DataLoader for LLM fine-tuning.

    Args:
        dataset: AfriScience dataset
        tokenizer: LLM tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        instruction_template: Custom instruction template
        shuffle: Whether to shuffle
        num_workers: Number of workers

    Returns:
        DataLoader for LLM fine-tuning
    """
    if instruction_template is None:
        instruction_template = "Translate the following {source_lang} text to {target_lang}:"

    collator = LLMCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        instruction_template=instruction_template,
    )

    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
