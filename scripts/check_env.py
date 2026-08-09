#!/usr/bin/env python
"""Phase 0 environment check.

Verifies the toolchain is installed and CUDA is available before you build.
Run:  uv run python scripts/check_env.py
"""
from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = [
    ("numpy", None),
    ("pandas", None),
    ("yaml", "pyyaml"),
    ("torch", None),
    ("torchaudio", None),
    ("faster_whisper", "faster-whisper"),
    ("pyannote.audio", "pyannote.audio"),
    ("opensmile", None),
    ("spacy", None),
    ("sentence_transformers", "sentence-transformers"),
    ("sklearn", "scikit-learn"),
    ("xgboost", None),
]

OK, WARN, FAIL = "\033[92mOK\033[0m", "\033[93mWARN\033[0m", "\033[91mFAIL\033[0m"


def check_import(module: str, pkg: str | None) -> bool:
    try:
        importlib.import_module(module)
        print(f"  [{OK}]   {module}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [{FAIL}] {module}  (install: {pkg or module})  -> {type(e).__name__}")
        return False


def main() -> int:
    print(f"\nPython {platform.python_version()} @ {sys.executable}\n")

    print("Packages:")
    results = [check_import(m, p) for m, p in REQUIRED]

    print("\nGPU / CUDA:")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  [{OK}]   CUDA available -> {torch.cuda.get_device_name(0)}")
            print(f"          torch {torch.__version__}, CUDA {torch.version.cuda}")
        else:
            print(f"  [{WARN}] CUDA NOT available. Set transcription.compute_type=int8 "
                  f"in config.yaml and expect slower runs.")
    except Exception as e:  # noqa: BLE001
        print(f"  [{FAIL}] could not query torch: {e}")

    print("\nspaCy model:")
    try:
        import spacy

        spacy.load("en_core_web_sm")
        print(f"  [{OK}]   en_core_web_sm loaded")
    except Exception:
        print(f"  [{WARN}] en_core_web_sm missing -> "
              f"run: uv run python -m spacy download en_core_web_sm")

    print("\nHuggingFace token (needed for pyannote diarization):")
    import os

    if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"):
        print(f"  [{OK}]   token found in environment")
    else:
        print(f"  [{WARN}] no HF_TOKEN in env. Accept model terms for "
              f"pyannote/speaker-diarization-3.1 and export HF_TOKEN=...")

    print()
    if all(results):
        print(f"[{OK}] Core packages import cleanly. You're ready for Phase 1.\n")
        return 0
    print(f"[{FAIL}] Some packages failed. Run: uv sync --extra app --extra dev\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
