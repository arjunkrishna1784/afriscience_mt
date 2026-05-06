"""Factory functions for creating API clients."""

from typing import Dict, List, Optional, Type

from .base import APIClient, APIConfig
from .openai_client import OpenAIClient
from .google_client import GoogleAIClient
from .together_client import TogetherAIClient


# Registry of available clients
CLIENT_REGISTRY: Dict[str, Type[APIClient]] = {
    "openai": OpenAIClient,
    "google": GoogleAIClient,
    "together": TogetherAIClient,
}

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "google": "gemini-1.5-pro",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}


def create_api_client(
    provider: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    **kwargs,
) -> APIClient:
    """
    Create an API client.

    Args:
        provider: API provider ("openai", "google", "together")
        model_name: Model name (uses default if not provided)
        api_key: API key (uses environment variable if not provided)
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional configuration options

    Returns:
        Configured API client

    Raises:
        ValueError: If provider is not supported
    """
    provider = provider.lower()

    if provider not in CLIENT_REGISTRY:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available providers: {list(CLIENT_REGISTRY.keys())}"
        )

    if model_name is None:
        model_name = DEFAULT_MODELS.get(provider)

    config = APIConfig(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **{k: v for k, v in kwargs.items() if k in APIConfig.__dataclass_fields__},
    )

    client_class = CLIENT_REGISTRY[provider]
    return client_class(config)


def get_available_clients() -> List[str]:
    """Get list of available API client providers."""
    return list(CLIENT_REGISTRY.keys())


def register_client(provider: str, client_class: Type[APIClient]) -> None:
    """
    Register a custom API client.

    Args:
        provider: Provider name
        client_class: Client class (must inherit from APIClient)
    """
    if not issubclass(client_class, APIClient):
        raise TypeError("client_class must inherit from APIClient")

    CLIENT_REGISTRY[provider] = client_class
