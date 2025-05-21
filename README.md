# CosmicPINN: Physics-Informed Neural Networks for Large-Scale Structure Formation

CosmicPINN is a **scientific-machine-learning** showcase that simulates the formation of cosmic large-scale structure by training a Physics-Informed Neural Network (PINN) to solve the coupled fluid and gravitational field equations that describe a matter-dominated universe.

The project is intentionally lightweight yet idiomatic, demonstrating:

* Clean, modular Python package layout (`cosmic_pinn/`).
* Automated experiment configuration with YAML.
* Reproducible environments via `requirements.txt`.
* Test-driven development with **pytest**.
* Type hints & docstrings for maintainability.
* Continuous integration ready (GitHub Actions template in `ci/`, optional).
* Simple plotting utilities for qualitative assessment.

<p align="center">
  <img src="docs/example_evolution.gif" width="600" alt="Evolution of density field">  
</p>

---

## 1. Quickstart

```bash
# Clone and install dependencies (preferably in a virtual env)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train the PINN (defaults are defined in config.yaml)
python -m cosmic_pinn.train

# Evaluate & visualise
python -m cosmic_pinn.evaluate --epoch best
```

All artifacts (trained weights, loss curves, and figures) are stored in `outputs/`.

---

## 2. Physical Model

We treat the matter content as an **inviscid, pressureless fluid** evolving in an expanding, Newtonian universe. In comoving coordinates \((x,y)\) and cosmic time \(t\) (with \(a=1\) for simplicity), the governing equations are:

1. Continuity  
\[\partial_t\,\delta + \nabla\cdot\mathbf{v}=0\]
2. Euler (2-D)  
\[\partial_t\,\mathbf{v} + (\mathbf{v}\cdot\nabla)\mathbf{v} + \nabla\phi = 0\]
3. Poisson  
\[\nabla^2\phi - \delta = 0\]

where \(\delta = (\rho-\bar{\rho})/\bar{\rho}\) is the overdensity, \(\mathbf{v} = (u,v)\) is the peculiar velocity, and \(\phi\) is the Newtonian potential.  

These PDEs are enforced **implicitly** by the PINN through automatic differentiation.

---

## 3. Project Structure

```
CosmicPINN/
├── cosmic_pinn/          # Python package
│   ├── __init__.py
│   ├── data.py           # Collocation sampling & IC generator
│   ├── equations.py      # PDE residuals
│   ├── pinn.py           # Network architecture
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Post-training visualisations
│   └── utils.py
├── tests/                # Pytest unit tests
│   ├── test_equations.py
│   └── test_pinn.py
├── outputs/              # Created at runtime (checkpoints, figs, logs)
├── requirements.txt
├── config.yaml           # Hyper-parameters
├── .gitignore
└── README.md
```

---

## 4. Configuration

Hyper-parameters are provided in `config.yaml` and can be overridden via the CLI. Example:

```bash
python -m cosmic_pinn.train \
    trainer.epochs=5000 \
    data.n_collocation=20000
```

---

## 5. Testing

Run unit tests with:

```bash
pytest -q
```

They include quick checks that the analytical trivial solution (all zeros) satisfies the residuals within tolerance.

---

## 6. License & Citation

This repository is released under the MIT license. If you use **CosmicPINN** in your research or portfolio, please cite appropriately.
