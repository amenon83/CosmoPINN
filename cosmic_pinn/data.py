"""Data sampling utilities for PINN training."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .utils import load_config


def sample_collocation(cfg: dict) -> torch.Tensor:
    """Sample interior collocation points uniformly in the domain."""
    n_col: int = cfg["data"]["n_collocation"]
    dom = cfg["physics"]["domain"]
    t = np.random.uniform(dom["t_min"], dom["t_max"], (n_col, 1))
    x = np.random.uniform(dom["x_min"], dom["x_max"], (n_col, 1))
    y = np.random.uniform(dom["y_min"], dom["y_max"], (n_col, 1))
    pts = np.hstack([t, x, y]).astype(np.float32)
    return torch.from_numpy(pts)


def sample_initial(cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (inputs, targets) on the t=0 surface according to ICs."""
    n_ic: int = cfg["data"]["n_initial"]
    dom = cfg["physics"]["domain"]
    ic = cfg["physics"]["initial_conditions"]

    x = np.random.uniform(dom["x_min"], dom["x_max"], (n_ic, 1))
    y = np.random.uniform(dom["y_min"], dom["y_max"], (n_ic, 1))
    t = np.zeros_like(x)
    inputs = np.hstack([t, x, y]).astype(np.float32)

    # Targets: delta, u, v, phi at t=0
    if ic["type"] == "gaussian":
        amp = ic["amplitude"]
        sigma = ic["sigma"]
        cx, cy = ic["center"]
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        delta0 = amp * np.exp(-r2 / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown IC type: {ic['type']}")

    targets = np.hstack([delta0, np.zeros_like(delta0), np.zeros_like(delta0), np.zeros_like(delta0)]).astype(np.float32)
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def prepare_datasets(cfg_path: str | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience helper to collect fresh collocation & IC samples."""
    cfg = load_config(cfg_path or "config.yaml")
    collocation_pts = sample_collocation(cfg)
    ic_inputs, ic_targets = sample_initial(cfg)
    return collocation_pts, ic_inputs, ic_targets 