"""Utility helpers for CosmicPINN."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config file and return a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def meshgrid_2d(xmin: float, xmax: float, ymin: float, ymax: float, n: int) -> torch.Tensor:
    """Return a 2D meshgrid of shape (n^2, 2)."""
    x = torch.linspace(xmin, xmax, n)
    y = torch.linspace(ymin, ymax, n)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    pts = torch.stack((X.flatten(), Y.flatten()), dim=1)
    return pts 