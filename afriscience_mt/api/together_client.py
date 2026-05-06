"""TogetherAI API client for hosted open models."""

from typing import Any, Dict, List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import APIClient, APIConfig


class TogetherAIClient(APIClient):
    """Client for TogetherAI API (Llama, Gemma, etc.)."""

    def _initialize_client(self) -> None:
        """Initialize TogetherAI client."""
        try:
            from together import Together
        except ImportError:
            raise ImportError(
                "together package not installed. "
                "Install with: pip install together"
            )

        self.client = Together(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(Exception),
    )
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Make TogetherAI API call.

        Args:
            messages: List of message dicts
            **kwargs: Additional arguments

        Returns:
            Generated response
        """
        generation_kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Add stop sequences if provided
        if "stop" in kwargs:
            generation_kwargs["stop"] = kwargs["stop"]

        response = self.client.chat.completions.create(**generation_kwargs)
        return response.choices[0].message.content

    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        Generate response with streaming.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments

        Yields:
            Response chunks
        """
        if self.client is None:
            self._initialize_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        self.rate_limiter.wait()

        stream = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.

        Returns:
            List of model info dicts
        """
        if self.client is None:
            self._initialize_client()

        models = self.client.models.list()
        return [
            {
                "id": model.id,
                "name": getattr(model, "display_name", model.id),
                "context_length": getattr(model, "context_length", None),
            }
            for model in models
        ]


# Model name mappings for TogetherAI
TOGETHER_MODEL_MAPPINGS = {
    # Llama models
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",

    # Gemma models
    "gemma-2-27b": "google/gemma-2-27b-it",
    "gemma-2-9b": "google/gemma-2-9b-it",
}


def get_together_model_id(model_name: str) -> str:
    """
    Get TogetherAI model ID from short name.

    Args:
        model_name: Short model name or full ID

    Returns:
        Full TogetherAI model ID
    """
    return TOGETHER_MODEL_MAPPINGS.get(model_name.lower(), model_name)
