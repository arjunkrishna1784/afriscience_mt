"""Base API client class."""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


@dataclass
class APIConfig:
    """Configuration for API clients."""

    provider: str  # "openai", "google", "together"
    model_name: str
    api_key: Optional[str] = None
    api_key_env_var: Optional[str] = None

    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 256
    top_p: float = 1.0

    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Additional settings
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_api_key(self) -> str:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key

        env_var = self.api_key_env_var
        if env_var is None:
            env_vars = {
                "openai": "OPENAI_API_KEY",
                "google": "GOOGLE_API_KEY",
                "together": "TOGETHER_API_KEY",
            }
            env_var = env_vars.get(self.provider)

        if env_var:
            key = os.environ.get(env_var)
            if key:
                return key

        raise ValueError(
            f"No API key found for {self.provider}. "
            f"Set {env_var} environment variable or provide api_key in config."
        )


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0

    def wait(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


class APIClient(ABC):
    """Abstract base class for API clients."""

    def __init__(self, config: APIConfig):
        """
        Initialize API client.

        Args:
            config: API configuration
        """
        self.config = config
        self.api_key = config.get_api_key()
        self.rate_limiter = RateLimiter(config.requests_per_minute)
        self.client = None

    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the underlying API client."""
        pass

    @abstractmethod
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Make a single API call.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation arguments

        Returns:
            Generated text response
        """
        pass

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation arguments

        Returns:
            Generated response
        """
        if self.client is None:
            self._initialize_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        self.rate_limiter.wait()
        return self._call_api(messages, **kwargs)

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt (same for all)
            **kwargs: Additional generation arguments

        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, system_prompt, **kwargs)
            responses.append(response)
        return responses

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        template: Optional[str] = None,
        glossary: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Translate text using the API.

        Args:
            text: Source text to translate
            source_lang: Source language name
            target_lang: Target language name
            template: Custom prompt template
            glossary: Optional glossary string
            examples: Optional list of example translations

        Returns:
            Translated text
        """
        # Build prompt
        if template:
            prompt = template.format(
                source_language=source_lang,
                target_language=target_lang,
                source_text=text,
                glossary=glossary or "",
                examples=self._format_examples(examples) if examples else "",
            )
        else:
            prompt = self._build_translation_prompt(
                text, source_lang, target_lang, glossary, examples
            )

        system_prompt = f"You are a professional translator specializing in scientific texts from {source_lang} to {target_lang}."

        response = self.generate(prompt, system_prompt)
        return self._extract_translation(response)

    def _build_translation_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Build a default translation prompt."""
        parts = []

        if examples:
            parts.append("Examples:")
            parts.append(self._format_examples(examples))
            parts.append("")

        if glossary:
            parts.append(f"Glossary ({source_lang} → {target_lang}):")
            parts.append(glossary)
            parts.append("")

        parts.append(f"Translate the following {source_lang} scientific text to {target_lang}:")
        parts.append("")
        parts.append(f"Source: {text}")
        parts.append("Translation:")

        return "\n".join(parts)

    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format examples for the prompt."""
        formatted = []
        for ex in examples:
            formatted.append(f"Source: {ex['source']}")
            formatted.append(f"Translation: {ex['target']}")
            formatted.append("")
        return "\n".join(formatted)

    def _extract_translation(self, response: str) -> str:
        """Extract translation from model response."""
        response = response.strip()

        # Handle "preamble:\n\n translation" pattern.
        # Models sometimes output a native-language header (e.g. "ga fassarar nan:\n\n")
        # before the actual translation.  Split on the first ":\n\n" and keep the
        # trailing part only when the prefix is short (≤100 chars), so we don't
        # accidentally discard multi-sentence translations that happen to contain ":\n".
        if ":\n\n" in response:
            pre, _, rest = response.partition(":\n\n")
            if len(pre) <= 100:
                response = rest.strip()

        # Strip common English output labels
        prefixes = ["Translation:", "translation:", "Output:", "output:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                break

        return response
