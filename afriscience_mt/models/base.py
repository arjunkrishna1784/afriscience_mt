"""Base model classes and interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class ModelConfig:
    """Configuration for translation models."""

    name: str
    model_type: str  # "seq2seq", "llm_open", "llm_closed"
    max_length: int = 256
    device: str = "auto"
    precision: str = "fp16"  # fp32, fp16, bf16

    # For local models
    cache_dir: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # For LoRA
    lora_config: Optional[Dict[str, Any]] = None

    # For API models
    api_provider: Optional[str] = None  # openai, google, together
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 256

    # Additional model-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseTranslationModel(ABC):
    """Abstract base class for translation models."""

    def __init__(self, config: ModelConfig):
        """
        Initialize model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        """Get the device to use."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def translate(
        self,
        texts: Union[str, List[str]],
        source_lang: str,
        target_lang: str,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Translate text(s).

        Args:
            texts: Input text or list of texts
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional generation arguments

        Returns:
            Translated text(s)
        """
        pass

    @abstractmethod
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 8,
        **kwargs,
    ) -> List[str]:
        """
        Translate a batch of texts.

        Args:
            texts: List of input texts
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Batch size for processing
            **kwargs: Additional generation arguments

        Returns:
            List of translated texts
        """
        pass

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get number of model parameters.

        Args:
            trainable_only: If True, only count trainable parameters

        Returns:
            Number of parameters
        """
        if self.model is None:
            return 0

        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Output path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

    def to(self, device: Union[str, torch.device]) -> "BaseTranslationModel":
        """
        Move model to device.

        Args:
            device: Target device

        Returns:
            Self
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        if self.model is not None:
            self.model.to(device)

        return self

    def eval(self) -> "BaseTranslationModel":
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self

    def train(self) -> "BaseTranslationModel":
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, device={self.device})"
