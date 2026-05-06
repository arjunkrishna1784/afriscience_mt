"""LLM model wrappers for Gemma and Llama."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .base import BaseTranslationModel, ModelConfig
from ..utils.language_utils import get_language_name


class LLMModel(BaseTranslationModel):
    """Base wrapper for LLM models used for translation."""

    def __init__(self, config: ModelConfig):
        """
        Initialize LLM model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.generation_config = {
            "max_new_tokens": config.max_tokens,
            "do_sample": False,
            "temperature": config.temperature,
            "pad_token_id": None,  # Set after loading tokenizer
        }

    def load(self) -> None:
        """Load model and tokenizer."""
        model_kwargs = {}

        # Set precision and quantization
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        elif self.config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.config.precision == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.config.precision == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16

        if self.config.cache_dir:
            model_kwargs["cache_dir"] = self.config.cache_dir

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            device_map="auto" if self.config.device == "auto" else None,
            **model_kwargs
        )

        if self.config.device != "auto":
            self.model.to(self.device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            cache_dir=self.config.cache_dir,
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id

    def _format_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        template: Optional[str] = None,
    ) -> str:
        """
        Format input text as a prompt.

        Args:
            text: Source text
            source_lang: Source language code
            target_lang: Target language code
            template: Optional custom template

        Returns:
            Formatted prompt
        """
        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)

        if template:
            return template.format(
                source_language=source_name,
                target_language=target_name,
                source_text=text,
            )

        # Default template
        return f"Translate the following {source_name} scientific text to {target_name}.\n\nSource: {text}\nTranslation:"

    def _apply_chat_template(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Apply chat template if available.

        Args:
            text: Source text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Formatted prompt with chat template
        """
        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)

        instruction = f"Translate the following {source_name} scientific text to {target_name}."
        user_message = f"{instruction}\n\n{text}"

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_message}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return self._format_prompt(text, source_lang, target_lang)

    def translate(
        self,
        texts: Union[str, List[str]],
        source_lang: str,
        target_lang: str,
        use_chat_template: bool = True,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Translate text(s).

        Args:
            texts: Input text or list of texts
            source_lang: Source language code
            target_lang: Target language code
            use_chat_template: Whether to use chat template
            **kwargs: Additional generation arguments

        Returns:
            Translated text(s)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        translations = self.translate_batch(
            texts, source_lang, target_lang,
            batch_size=len(texts),
            use_chat_template=use_chat_template,
            **kwargs
        )

        return translations[0] if single_input else translations

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 4,
        use_chat_template: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Translate a batch of texts.

        Args:
            texts: List of input texts
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Batch size for processing
            use_chat_template: Whether to use chat template
            **kwargs: Additional generation arguments

        Returns:
            List of translated texts
        """
        translations = []
        generation_config = {**self.generation_config, **kwargs}

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Format prompts
            if use_chat_template:
                prompts = [
                    self._apply_chat_template(text, source_lang, target_lang)
                    for text in batch_texts
                ]
            else:
                prompts = [
                    self._format_prompt(text, source_lang, target_lang)
                    for text in batch_texts
                ]

            # Tokenize
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config,
                )

            # Decode only the new tokens
            for j, output in enumerate(outputs):
                input_length = inputs["input_ids"][j].shape[0]
                new_tokens = output[input_length:]
                translation = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                translations.append(translation.strip())

        return translations


class GemmaModel(LLMModel):
    """Wrapper for Gemma models."""

    def load(self) -> None:
        """Load Gemma model with appropriate settings."""
        super().load()

        # Gemma-specific settings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


class LlamaModel(LLMModel):
    """Wrapper for Llama models."""

    def load(self) -> None:
        """Load Llama model with appropriate settings."""
        super().load()

        # Llama-specific settings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"


def load_llm_model(
    model_name: str,
    model_type: Optional[str] = None,
    device: str = "auto",
    precision: str = "bf16",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> LLMModel:
    """
    Load an LLM for translation.

    Args:
        model_name: Model name or path
        model_type: Model type ("gemma", "llama"). Auto-detected if None.
        device: Device to use
        precision: Precision (fp32, fp16, bf16)
        load_in_4bit: Use 4-bit quantization
        load_in_8bit: Use 8-bit quantization
        cache_dir: Cache directory
        **kwargs: Additional config options

    Returns:
        Loaded model wrapper
    """
    # Auto-detect model type
    if model_type is None:
        model_name_lower = model_name.lower()
        if "gemma" in model_name_lower:
            model_type = "gemma"
        elif "llama" in model_name_lower:
            model_type = "llama"
        else:
            model_type = "llm"  # Generic LLM

    config = ModelConfig(
        name=model_name,
        model_type=model_type,
        device=device,
        precision=precision,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        cache_dir=cache_dir,
        extra=kwargs,
    )

    if model_type == "gemma":
        model = GemmaModel(config)
    elif model_type == "llama":
        model = LlamaModel(config)
    else:
        model = LLMModel(config)

    model.load()
    return model


def load_lora_model(
    base_model_name: str,
    lora_path: Union[str, Path],
    model_type: Optional[str] = None,
    device: str = "auto",
    load_in_4bit: bool = True,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> LLMModel:
    """
    Load an LLM with LoRA adapters.

    Args:
        base_model_name: Base model name
        lora_path: Path to LoRA adapter weights
        model_type: Model type ("gemma", "llama")
        device: Device to use
        load_in_4bit: Use 4-bit quantization
        cache_dir: Cache directory
        **kwargs: Additional config options

    Returns:
        Model with LoRA adapters loaded
    """
    from peft import PeftModel

    # Load base model
    config = ModelConfig(
        name=base_model_name,
        model_type=model_type or "llm",
        device=device,
        load_in_4bit=load_in_4bit,
        cache_dir=cache_dir,
        lora_config={"path": str(lora_path)},
        extra=kwargs,
    )

    if model_type == "gemma":
        model_wrapper = GemmaModel(config)
    elif model_type == "llama":
        model_wrapper = LlamaModel(config)
    else:
        model_wrapper = LLMModel(config)

    model_wrapper.load()

    # Load LoRA adapters
    model_wrapper.model = PeftModel.from_pretrained(
        model_wrapper.model,
        lora_path,
    )

    return model_wrapper
