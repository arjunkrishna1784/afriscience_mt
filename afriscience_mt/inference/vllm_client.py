"""vLLM client for fast batched LLM inference.

Drop-in replacement for LocalLLMClient with significantly higher throughput
via PagedAttention and continuous batching. Supports LoRA adapters.

Install: pip install vllm
"""

from typing import Any, Dict, List, Optional


class VLLMClient:
    """Client for running HuggingFace LLMs locally via vLLM.

    Advantages over LocalLLMClient:
    - True batched inference (all prompts processed together, not one-by-one)
    - PagedAttention for efficient KV-cache memory management
    - Continuous batching for high throughput
    - Native LoRA adapter support
    - No bitsandbytes dependency (vLLM handles quantization internally)
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        lora_path: Optional[str] = None,
        max_new_tokens: int = 512,
        trust_remote_code: bool = True,
        download_dir: Optional[str] = None,
    ):
        """
        Initialize vLLM client.

        Args:
            model_name: HuggingFace model ID or local path
                        (e.g., "masakhane/AfriqueLlama-3.2-1B-Instruct")
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Max sequence length (auto-detected from config if None)
            dtype: Model dtype ("auto", "float16", "bfloat16", "float32")
            quantization: Quantization method (None, "awq", "gptq", "squeezellm")
            lora_path: Path to LoRA adapter weights (enables LoRA mode)
            max_new_tokens: Default max tokens to generate
            trust_remote_code: Whether to trust remote code in model configs
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install with:\n"
                "  pip install vllm\n"
                "Note: vLLM requires CUDA and Linux/WSL2."
            )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.lora_path = lora_path
        self._lora_request = None
        self._SamplingParams = SamplingParams

        llm_kwargs: Dict[str, Any] = dict(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if quantization is not None:
            llm_kwargs["quantization"] = quantization
        if lora_path is not None:
            llm_kwargs["enable_lora"] = True
        if download_dir is not None:
            llm_kwargs["download_dir"] = download_dir

        print(f"Loading model with vLLM: {model_name}")
        if lora_path:
            print(f"  LoRA adapter: {lora_path}")
        self.llm = LLM(**llm_kwargs)

        if lora_path is not None:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest("adapter", 1, str(lora_path))

        # Cache tokenizer reference for chat template application
        self._tokenizer = self.llm.get_tokenizer()

        # Stash the model's hard context limit so we can pass truncate_prompt_tokens
        # to SamplingParams and avoid crashing on documents that exceed the window.
        try:
            self.max_model_len = self.llm.llm_engine.model_config.max_model_len
        except AttributeError:
            self.max_model_len = None

        print("Model loaded successfully!")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_chat_model(self) -> bool:
        """Heuristic: model name suggests instruction-tuning."""
        lower = self.model_name.lower()
        return any(
            tok in lower
            for tok in ["instruct", "chat", "-it", "assistant", "alpaca", "vicuna"]
        )

    def _format_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Apply chat/instruction template if one exists, else return prompt as-is.

        Strategy (in order):
        1. Tokenizer chat template — used whenever the loaded tokenizer ships one,
           regardless of whether the model name looks like an instruct model.
           This correctly handles models such as McGill-NLP/AfriqueLlama-8B or
           McGill-NLP/AfriqueQwen-8B that are fine-tuned with a specific format
           but do not have "instruct" in their names.
        2. Name-heuristic fallback — if no tokenizer template but the model name
           suggests instruction tuning, concatenate system + user prompts.
        3. Plain prompt — true base models with no template at all.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 1. Try tokenizer chat template (works for both instruct and fine-tuned base models)
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # 2. Name-heuristic fallback for models without a stored chat template
        if self._is_chat_model():
            if system_prompt:
                return f"{system_prompt}\n\n{prompt}"
            return prompt

        # 3. True base model — return plain prompt as-is
        return prompt

    def _make_sampling_params(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ):
        n_new = max_new_tokens or self.max_new_tokens

        # If the model has a hard context limit, tell vLLM to truncate prompts
        # that are too long rather than raising an error.  This is a safety
        # net — with max_model_len=8192 and max_new_tokens=1024 the threshold
        # is 7168 tokens, which covers all current documents even in 1-shot
        # mode.  generate_batch() logs a WARNING if truncation actually fires.
        truncate: Optional[int] = None
        if self.max_model_len is not None:
            truncate = max(1, self.max_model_len - n_new)

        kwargs: Dict[str, Any] = dict(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            max_tokens=n_new,
        )
        # vLLM <=0.7 accepted truncate_prompt_tokens on SamplingParams;
        # 0.8+ removed it (max_model_len at the engine level handles the
        # cap, and over-budget prompts now error at generate-time).
        # Probe the SDK so we work on either era.
        if truncate is not None and self._supports_truncate_kwarg():
            kwargs["truncate_prompt_tokens"] = truncate

        return self._SamplingParams(**kwargs)

    def _supports_truncate_kwarg(self) -> bool:
        """True if SamplingParams accepts truncate_prompt_tokens (vLLM <=0.7)."""
        cached = getattr(self, "_trunc_supported", None)
        if cached is not None:
            return cached
        import inspect
        try:
            params = inspect.signature(self._SamplingParams).parameters
            cached = "truncate_prompt_tokens" in params
        except (TypeError, ValueError):
            cached = False
        self._trunc_supported = cached
        return cached

    # ------------------------------------------------------------------
    # Public API — compatible with LocalLLMClient
    # ------------------------------------------------------------------

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
        Generate text for a single prompt.

        Args:
            prompt: Input prompt (plain text or pre-formatted)
            system_prompt: Optional system message for chat models
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p nucleus sampling
            do_sample: Enable sampling (overrides greedy decoding)

        Returns:
            Generated text (new tokens only)
        """
        results = self.generate_batch(
            prompts=[prompt],
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            show_progress=False,
        )
        return results[0]

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Generate text for multiple prompts in a single batched vLLM call.

        This is the main performance advantage over LocalLLMClient: all prompts
        are sent to vLLM at once and processed with continuous batching.

        Args:
            prompts: List of input prompts
            system_prompt: Optional system message applied to all prompts
            max_new_tokens: Max tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Top-p nucleus sampling
            do_sample: Enable sampling
            show_progress: Ignored (vLLM handles progress internally)

        Returns:
            List of generated texts, one per input prompt
        """
        formatted = [self._format_prompt(p, system_prompt) for p in prompts]
        sampling_params = self._make_sampling_params(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        # Warn when any prompt in the batch will actually be truncated.
        # We use the tokenizer for an exact count (not a char-based proxy) so
        # the warning is reliable and the per-prompt breakdown is logged.
        trunc_limit = getattr(sampling_params, "truncate_prompt_tokens", None)
        if trunc_limit is not None:
            try:
                import logging as _logging
                _log = _logging.getLogger(__name__)
                token_counts = [
                    len(self._tokenizer.encode(p)) for p in formatted
                ]
                truncated = [(i, c) for i, c in enumerate(token_counts) if c > trunc_limit]
                if truncated:
                    _log.warning(
                        "vLLM prompt truncation: %d/%d prompts exceed "
                        "truncate_prompt_tokens=%d and will be LEFT-TRUNCATED "
                        "(max_model_len=%d). Longest: %d tokens. "
                        "Consider raising max_model_len in the config.",
                        len(truncated), len(formatted), trunc_limit,
                        self.max_model_len,
                        max(c for _, c in truncated),
                    )
            except Exception:
                pass  # tokenization failure must never break inference

        generate_kwargs: Dict[str, Any] = {"sampling_params": sampling_params}
        if self._lora_request is not None:
            generate_kwargs["lora_request"] = self._lora_request

        outputs = self.llm.generate(formatted, **generate_kwargs)
        return [o.outputs[0].text.strip() for o in outputs]

    # ------------------------------------------------------------------
    # Convenience aliases (API-client compatibility)
    # ------------------------------------------------------------------

    def complete(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """API-compatible completion returning a dict with 'text' key."""
        return {"text": self.generate(prompt, **kwargs), "model": self.model_name}

    def translate(self, prompt: str) -> str:
        """Alias for generate (used by ICLRunner compatibility)."""
        return self.generate(prompt)

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)
