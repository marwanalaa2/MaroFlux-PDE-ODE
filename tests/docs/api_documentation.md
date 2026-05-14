# MaroFlux API Documentation

## Overview

The MaroFlux codebase implements the Unified MaroFlux Equation (UME) — a coupled PDE-ODE system for simulating autonomous immunomodulatory biomaterials.

## Main Modules

### `maroflux_solver_complete.py`

The main solver class: `MaroFluxSolver`.

#### `MaroFluxSolver(params)`
Constructor.
- **params** (`dict`): Parameter dictionary (loaded from YAML).

#### `run()`
Runs the full simulation from t=0 to t_end.

#### Attributes after run:
- `time_points`: array of output times.
- `M1_history`, `M2_history`: macrophage densities over time.
- `ion_release`: dict of ion release curves.

### `ume_pde_solver.py`

Class: `MaroFluxPDESolver`

Solves the PDE component (reaction-diffusion-convection) using FEniCS.

### `ume_ode_solver.py`

Class: `MacrophageODESolver`

Solves the macrophage ODE system at each spatial node.

### `coupling_manager.py`

Implements Strang operator splitting to couple PDE and ODE solves.

### `sensitivity_analysis.py`

Performs Sobol' global sensitivity analysis using SALib.

### `bifurcation_analysis.py`

Computes equilibrium curves and maps the Safe Operational Space (SOS).

### `visualization.py`

Generates publication-quality figures from simulation output.

## Parameter Files

- `parameters/parameter_set.yaml`: Complete UME parameters (Table 3.1).
- `parameters/muscle_parameters.yaml`: Muscle model parameters (Table 4.1).

## Tests

Run with `pytest` from the repository root:
```bash
pytest tests/
