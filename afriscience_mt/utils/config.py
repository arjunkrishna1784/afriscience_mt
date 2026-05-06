"""Configuration management using OmegaConf."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os

from omegaconf import OmegaConf, DictConfig


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file
        overrides: Optional dictionary of overrides to apply

    Returns:
        OmegaConf DictConfig object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    # Apply overrides if provided
    if overrides:
        override_config = OmegaConf.create(overrides)
        config = OmegaConf.merge(config, override_config)

    # Resolve interpolations
    OmegaConf.resolve(config)

    return config


def merge_configs(*configs: Union[DictConfig, Dict]) -> DictConfig:
    """
    Merge multiple configurations, with later configs taking precedence.

    Args:
        *configs: Variable number of configs to merge

    Returns:
        Merged DictConfig
    """
    result = OmegaConf.create({})

    for config in configs:
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        result = OmegaConf.merge(result, config)

    return result


def load_experiment_config(
    experiment_type: str,
    config_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Load experiment-specific configuration with base config.

    Args:
        experiment_type: Type of experiment (e.g., "lora_finetuning", "icl_sentence")
        config_dir: Directory containing config files
        overrides: Optional overrides

    Returns:
        Merged configuration
    """
    if config_dir is None:
        from afriscience_mt import DEFAULT_CONFIG_DIR
        config_dir = DEFAULT_CONFIG_DIR

    config_dir = Path(config_dir)

    # Load base config
    base_config = load_config(config_dir / "base.yaml")

    # Load experiment-specific config
    experiment_config_path = config_dir / "experiments" / f"{experiment_type}.yaml"
    if experiment_config_path.exists():
        experiment_config = load_config(experiment_config_path)
    else:
        experiment_config = OmegaConf.create({})

    # Merge configs
    config = merge_configs(base_config, experiment_config)

    # Apply overrides
    if overrides:
        config = merge_configs(config, overrides)

    return config


def load_model_config(
    model_type: str,
    config_dir: Optional[Path] = None
) -> DictConfig:
    """
    Load model-specific configuration.

    Args:
        model_type: Type of model ("seq2seq", "llm_open", "llm_closed")
        config_dir: Directory containing config files

    Returns:
        Model configuration
    """
    if config_dir is None:
        from afriscience_mt import DEFAULT_CONFIG_DIR
        config_dir = DEFAULT_CONFIG_DIR

    config_dir = Path(config_dir)
    model_config_path = config_dir / "models" / f"{model_type}.yaml"

    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    return load_config(model_config_path)


def load_prompt_templates(
    template_type: str = "sentence",
    config_dir: Optional[Path] = None
) -> DictConfig:
    """
    Load prompt templates configuration.

    Args:
        template_type: Type of templates ("sentence" or "document")
        config_dir: Directory containing config files

    Returns:
        Prompt templates configuration
    """
    if config_dir is None:
        from afriscience_mt import DEFAULT_CONFIG_DIR
        config_dir = DEFAULT_CONFIG_DIR

    config_dir = Path(config_dir)
    templates_path = config_dir / "prompts" / f"{template_type}_templates.yaml"

    if not templates_path.exists():
        raise FileNotFoundError(f"Templates not found: {templates_path}")

    return load_config(templates_path)


def load_evaluation_prompts(config_dir: Optional[Path] = None) -> DictConfig:
    """
    Load GPT-4o evaluation prompts.

    Args:
        config_dir: Directory containing config files

    Returns:
        Evaluation prompts configuration
    """
    if config_dir is None:
        from afriscience_mt import DEFAULT_CONFIG_DIR
        config_dir = DEFAULT_CONFIG_DIR

    config_dir = Path(config_dir)
    prompts_path = config_dir / "prompts" / "evaluation_prompts.yaml"

    if not prompts_path.exists():
        raise FileNotFoundError(f"Evaluation prompts not found: {prompts_path}")

    return load_config(prompts_path)


def save_config(config: DictConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        OmegaConf.save(config, f)


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf config to plain dictionary."""
    return OmegaConf.to_container(config, resolve=True)


def get_api_key(service: str) -> Optional[str]:
    """
    Get API key from environment variable.

    Args:
        service: Service name ("openai", "google", "together")

    Returns:
        API key or None if not found
    """
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "together": "TOGETHER_API_KEY",
    }

    env_var = env_vars.get(service.lower())
    if env_var:
        return os.environ.get(env_var)

    return None
