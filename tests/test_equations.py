import torch

from cosmic_pinn.equations import trivial_solution_residual


def test_trivial_solution_residual_zero():
    # sample random points
    pts = torch.rand(128, 3)  # (t, x, y)
    res = trivial_solution_residual(pts)
    max_abs = res.abs().max().item()
    assert max_abs < 1e-6, f"Residuals not near zero: max={max_abs}" 