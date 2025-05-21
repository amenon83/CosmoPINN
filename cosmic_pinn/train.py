"""Training script for CosmicPINN.

Usage:
    python -m cosmic_pinn.train [--config path/to/config.yaml]

The script saves checkpoints and loss curves in `outputs/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from .data import prepare_datasets
from .pinn import CosmicPINN
from .utils import load_config, set_seed


def main(cfg: Dict):
    set_seed()

    device = torch.device(cfg["trainer"].get("device", "cpu"))

    # Prepare datasets (CPU tensors)
    collocation_pts, ic_inputs, ic_targets = prepare_datasets()

    # Send IC tensors to device once (they're small)
    ic_inputs = ic_inputs.to(device)
    ic_targets = ic_targets.to(device)

    # Instantiate model
    net_cfg = cfg["network"]
    model = CosmicPINN(net_cfg["layers"], net_cfg.get("activation", "tanh")).to(device)

    lr = float(cfg["trainer"]["lr"])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    num_epochs = cfg["trainer"]["epochs"]
    print_every = cfg["trainer"].get("print_every", 100)
    ckpt_every = cfg["trainer"].get("checkpoint_every", 500)

    pbar = trange(1, num_epochs + 1, desc="Training")
    for epoch in pbar:
        # Resample collocation points each epoch to avoid overfitting
        collocation_pts = prepare_datasets()[0].to(device)

        loss = model.total_loss(collocation_pts, ic_inputs, ic_targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % print_every == 0 or epoch == 1:
            pbar.set_postfix(loss=f"{loss.item():.3e}")

        if epoch % ckpt_every == 0 or epoch == num_epochs:
            ckpt_path = out_dir / f"model_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimiser.state_dict(),
                "loss": loss.item(),
            }, ckpt_path)

    print("Training completed ✨")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CosmicPINN")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config) 