"""Shared helpers: YAML configs, seeding, device selection, environment report."""

from __future__ import annotations

import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``. Dicts merge, everything else replaces."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. A top-level ``_base_`` key names a parent config, relative to this file."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        cfg = deep_update(load_config(path.parent / base), cfg)
    return cfg


def _parse_value(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        # PyYAML follows YAML 1.1, where "3e-4" is a string (a float needs a dot and a signed exponent).
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply command-line overrides like ``train.lr=3e-4``. Values are parsed as YAML."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' is not of the form key=value")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(raw)
    return cfg


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch on all devices.

    ``deterministic=True`` asks CUDA for repeatable kernels at a speed cost. It must run
    before the first CUDA operation because cuBLAS reads its workspace setting once.
    MPS has no equivalent guarantee.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds the CPU, CUDA and MPS generators
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(preference: str = "auto") -> torch.device:
    """``auto`` picks cuda, then mps, then cpu. An explicit choice that is unavailable raises."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available (x86 Python under Rosetta?)")
    return device


def env_report() -> dict[str, Any]:
    """Versions and hardware, printed at the start of every run and saved with results."""
    import torchvision

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "auto_device": str(pick_device()),
    }
    if torch.cuda.is_available():
        report["cuda_device"] = torch.cuda.get_device_name(0)
        report["cuda_runtime"] = torch.version.cuda
    return report