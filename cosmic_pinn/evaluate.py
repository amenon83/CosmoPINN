"""Post-training evaluation script.

Generates 2D density field snapshots and saves them to `outputs/figures/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .pinn import CosmicPINN
from .utils import load_config, meshgrid_2d


def load_checkpoint(model: CosmicPINN, ckpt_path: Path) -> None:
    data = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(data["model_state"])


def main(cfg: dict, epoch: int | str = "best") -> None:
    # Create evaluation grid
    dom = cfg["physics"]["domain"]
    grid = meshgrid_2d(dom["x_min"], dom["x_max"], dom["y_min"], dom["y_max"], n=100)
    n_pts = grid.shape[0]

    device = torch.device("cpu")

    # For multiple timesteps, but here we'll sample t=cfg["physics"]["domain"]["t_max"]
    t_final = cfg["physics"]["domain"]["t_max"]
    ts = torch.full((n_pts, 1), t_final)
    inputs = torch.cat([ts, grid], dim=1).to(device)

    # Build model
    net_cfg = cfg["network"]
    model = CosmicPINN(net_cfg["layers"], net_cfg.get("activation", "tanh")).to(device)

    # Resolve checkpoint path
    if epoch == "best":
        # just pick latest
        ckpts = sorted(Path("outputs").glob("model_epoch_*.pt"))
        ckpt_path = ckpts[-1]
    else:
        ckpt_path = Path("outputs") / f"model_epoch_{epoch}.pt"
    load_checkpoint(model, ckpt_path)
    model.eval()

    with torch.no_grad():
        preds = model(inputs)
    delta = preds[:, 0].reshape(100, 100).cpu()

    # Plot
    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.imshow(delta.T, origin="lower", extent=[dom["x_min"], dom["x_max"], dom["y_min"], dom["y_max"]])
    plt.colorbar(label=r"$\delta$ (overdensity)")
    plt.title("Density field at t = {:.2f}".format(t_final))
    plt.tight_layout()
    plt.savefig(fig_dir / f"density_epoch_{epoch}.png", dpi=200)
    print(f"Figure saved to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CosmicPINN model")
    parser.add_argument("--epoch", default="best", help="Checkpoint epoch to load (or 'best')")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg, epoch=args.epoch) 