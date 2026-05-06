"""Reproducibility utilities for experiment tracking."""

import random
import os
import sys
import platform
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for other libraries
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get system and library version information for reproducibility.

    Returns:
        Dictionary with version and system information
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {},
    }

    # Core packages
    try:
        import torch
        info["packages"]["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_devices"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    try:
        import transformers
        info["packages"]["transformers"] = transformers.__version__
    except ImportError:
        pass

    try:
        import peft
        info["packages"]["peft"] = peft.__version__
    except ImportError:
        pass

    try:
        import datasets
        info["packages"]["datasets"] = datasets.__version__
    except ImportError:
        pass

    try:
        import sacrebleu
        info["packages"]["sacrebleu"] = sacrebleu.__version__
    except ImportError:
        pass

    try:
        import comet
        info["packages"]["comet"] = comet.__version__
    except ImportError:
        pass

    try:
        import numpy
        info["packages"]["numpy"] = numpy.__version__
    except ImportError:
        pass

    try:
        import pandas
        info["packages"]["pandas"] = pandas.__version__
    except ImportError:
        pass

    try:
        import omegaconf
        info["packages"]["omegaconf"] = omegaconf.__version__
    except ImportError:
        pass

    return info


def get_git_info() -> Optional[Dict[str, str]]:
    """
    Get git repository information if available.

    Returns:
        Dictionary with git commit hash and branch, or None
    """
    try:
        import subprocess

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        # Check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        return {
            "commit": commit,
            "branch": branch,
            "dirty": len(status) > 0,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def create_experiment_manifest(
    experiment_name: str,
    config: Dict[str, Any],
    seed: int = 42,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a complete experiment manifest for reproducibility.

    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        seed: Random seed used
        additional_info: Any additional information to include

    Returns:
        Complete experiment manifest
    """
    manifest = {
        "experiment_name": experiment_name,
        "seed": seed,
        "config": config,
        "environment": get_reproducibility_info(),
        "git": get_git_info(),
    }

    if additional_info:
        manifest["additional_info"] = additional_info

    return manifest
