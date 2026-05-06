"""DCS-MT: machine translation framework for African scientific text."""

from pathlib import Path

__version__ = "0.1.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
