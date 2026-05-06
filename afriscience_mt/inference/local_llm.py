"""Local LLM client for GPU-based inference."""

import os
from typing import List, Optional, Dict, Any, Union
import torch
from tqdm import tqdm


class LocalLLMClient:
    """Client for running LLMs locally on GPU."""

    def __init__(
        self,
        model_name: str,
        quantization: str = "4bit",
        batch_size: int = 1,
        max_new_tokens: int = 512,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize local LLM client.

        Args:
            model_name: HuggingFace model name (e.g., "google/gemma-3-4b-it")
            quantization: Quantization method ("none", "4bit", "8bit")
            batch_size: Batch size for inference
            max_new_tokens: Maximum tokens to generate
            device: Device to use (auto-detected if None)
            torch_dtype: Torch dtype (auto-detected if None)
        """
        self.model_name = model_name
        self.quantization = quantization
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Auto-detect dtype
        if torch_dtype is None:
            if torch.cuda.is_available():
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer with appropriate configuration."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}, Quantization: {self.quantization}")

        # Configure quantization
        quantization_config = None
        load_in_4bit = False
        load_in_8bit = False

        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_in_4bit = True
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            load_in_8bit = True

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        elif self.device == "cuda":
            model_kwargs["torch_dtype"] = self.torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Move to device if not using device_map
        if self.device != "cuda" or "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """
        Generate text from a single prompt.

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt (ignored for most local models)
            max_new_tokens: Max tokens to generate (uses default if None)
            temperature: Sampling temperature
            top_p: Top-p sampling
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated text (response only, not including prompt)
        """
        # Ensure max_new_tokens is an integer
        if max_new_tokens is not None:
            max_new_tokens = int(max_new_tokens)
        else:
            max_new_tokens = self.max_new_tokens

        # Handle chat format for instruction-tuned models
        if self._is_chat_model():
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device)
                inputs = {"input_ids": inputs}
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

        # Generate
        with torch.no_grad():
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if temperature > 0 and do_sample:
                generation_config.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                })
            else:
                generation_config["do_sample"] = False

            outputs = self.model.generate(
                **inputs if isinstance(inputs, dict) else {"input_ids": inputs},
                **generation_config,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1] if isinstance(inputs, dict) else inputs.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            show_progress: Whether to show progress bar

        Returns:
            List of generated texts
        """
        results = []
        iterator = tqdm(prompts, desc="Generating") if show_progress else prompts

        for prompt in iterator:
            response = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results.append(response)

        return results

    def _is_chat_model(self) -> bool:
        """Check if model is instruction-tuned (chat) model."""
        model_lower = self.model_name.lower()
        chat_indicators = [
            "instruct", "chat", "-it", "conversational",
            "assistant", "alpaca", "vicuna"
        ]
        return any(indicator in model_lower for indicator in chat_indicators)

    def __call__(self, prompt: str, **kwargs) -> str:
        """Allow calling client directly as a function."""
        return self.generate(prompt, **kwargs)

    def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        API-compatible completion method.

        Returns dict with 'text' key for compatibility with API clients.
        """
        response = self.generate(prompt, **kwargs)
        return {"text": response, "model": self.model_name}

    def translate(self, prompt: str) -> str:
        """Translate using the model (alias for generate)."""
        return self.generate(prompt)
