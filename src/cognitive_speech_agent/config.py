"""Config loader. Single source of truth = config.yaml at repo root.

Importing this module also loads the repo-root .env into the environment (once),
so API keys like OPENAI_API_KEY / HF_TOKEN are available without manually
`source`-ing .env in every terminal. Real environment variables win over .env.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=value per line. Does not overwrite real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)          # setdefault -> real env wins


# load .env as soon as the package config is imported (i.e. every entry point)
_load_dotenv(REPO_ROOT / ".env")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the project config as a dict."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)