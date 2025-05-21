import torch

from cosmic_pinn.pinn import CosmicPINN


def test_pinn_forward_shape():
    model = CosmicPINN([3, 32, 32, 4])
    pts = torch.rand(10, 3)
    out = model(pts)
    assert out.shape == (10, 4) 