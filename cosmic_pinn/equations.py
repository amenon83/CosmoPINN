"""Physics equations and their residuals for the 2-D cosmological fluid.

The PINN predicts four fields as functions of (t, x, y):
  delta  – over-density (scalar)
  u      – x-component of peculiar velocity
  v      – y-component of peculiar velocity
  phi    – gravitational potential

The governing equations (assuming a=1) are:
  Continuity:   d/dt(delta) + du/dx + dv/dy = 0
  Euler-x:      du/dt + u du/dx + v du/dy + dphi/dx = 0
  Euler-y:      dv/dt + u dv/dx + v dv/dy + dphi/dy = 0
  Poisson:      d2phi/dx2 + d2phi/dy2 − delta = 0

All derivatives are obtained via automatic differentiation.
"""

from __future__ import annotations

from typing import Tuple

import torch

Tensor = torch.Tensor


def gradients(y: Tensor, x: Tensor, order: int = 1) -> Tensor:
    """Compute *total* derivatives of `y` w.r.t. `x`.

    Parameters
    ----------
    y : Tensor
        Scalar or vector valued function evaluated at `x`.
    x : Tensor
        Inputs with `requires_grad=True`.
    order : int, default=1
        If 1, returns dy/dx. If 2, returns the Hessian diagonal (second derivs).
    """
    if order not in (1, 2):
        raise ValueError("Only first and second order derivatives supported.")

    grad = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        grad = torch.zeros_like(x, requires_grad=True)

    if order == 1:
        return grad

    # Second order: loop over each dimension
    grads2 = []
    for i in range(x.shape[1]):
        # Take gradient of grad[:, i] w.r.t. x[:, i]
        grad2 = torch.autograd.grad(
            grad[:, i],
            x,
            grad_outputs=torch.ones_like(grad[:, i]),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad2 is None:
            grad2 = torch.zeros_like(x, requires_grad=True)
        grad2 = grad2[:, i].unsqueeze(1)  # second derivative wrt that coordinate
        grads2.append(grad2)
    return torch.cat(grads2, dim=1)


def residuals(
    inputs: Tensor,  # (N, 3)
    outputs: Tensor,  # (N, 4)
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return PDE residuals for a batch of collocation points.

    The tensors are shaped (N, 1) each and correspond to the four equations.
    """
    # Split outputs
    delta, u, v, phi = torch.split(outputs, 1, dim=1)  # each (N,1)

    t = inputs[:, [0]].requires_grad_(True)
    x = inputs[:, [1]].requires_grad_(True)
    y = inputs[:, [2]].requires_grad_(True)

    # Re-compute outputs with gradient tracking linked to (t,x,y)
    # We must concat and detach from original gradient tracking to avoid errors.
    coords = torch.cat([t, x, y], dim=1)

    delta = delta.detach().clone().requires_grad_(True)
    u = u.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    phi = phi.detach().clone().requires_grad_(True)

    # First order derivatives
    ddelta_dt = gradients(delta, coords)[:, 0:1]
    du_dt = gradients(u, coords)[:, 0:1]
    dv_dt = gradients(v, coords)[:, 0:1]

    du_dx = gradients(u, coords)[:, 1:2]
    du_dy = gradients(u, coords)[:, 2:3]
    dv_dx = gradients(v, coords)[:, 1:2]
    dv_dy = gradients(v, coords)[:, 2:3]
    dphi_dx = gradients(phi, coords)[:, 1:2]
    dphi_dy = gradients(phi, coords)[:, 2:3]

    # Second order derivatives for Poisson
    d2phi_dx2 = gradients(dphi_dx, coords)[:, 1:2]
    d2phi_dy2 = gradients(dphi_dy, coords)[:, 2:3]

    # Continuity residual
    r_cont = ddelta_dt + du_dx + dv_dy

    # Euler residuals
    r_mom_x = du_dt + u * du_dx + v * du_dy + dphi_dx
    r_mom_y = dv_dt + u * dv_dx + v * dv_dy + dphi_dy

    # Poisson residual
    r_poisson = d2phi_dx2 + d2phi_dy2 - delta

    return r_cont, r_mom_x, r_mom_y, r_poisson


def trivial_solution_residual(points: Tensor) -> Tensor:
    """Utility used in tests: evaluate residuals for the analytic trivial solution.

    For delta = 0, u = 0, v = 0, phi = 0, all residuals should vanish.
    """
    pts = points.clone().detach().requires_grad_(True)
    outputs = torch.zeros(pts.shape[0], 4, dtype=pts.dtype, device=pts.device, requires_grad=True)
    res = residuals(pts, outputs)
    return torch.cat(res, dim=1)  # (N,4) 