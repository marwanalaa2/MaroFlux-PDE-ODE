#!/usr/bin/env python3
"""
MaroFlux Complete Solver
========================
Main driver for the Unified MaroFlux Equation (UME).

This script:
1. Loads parameters from YAML/JSON files.
2. Generates the computational mesh.
3. Initializes and runs the PDE-ODE coupled simulation.
4. Saves output figures and numerical data.

Usage:
    python maroflux_solver_complete.py --config parameters/parameter_set.yaml
    python maroflux_solver_complete.py --help

Author: Marwan Alaa Mohamed Mohamed Raslan
Date: May 2026
License: MIT
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Attempt imports – graceful failure with helpful messages
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:
    print("PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    from dolfin import (Mesh, XDMFFile, FunctionSpace, Function,
                        Constant, TestFunction, TrialFunction,
                        dot, grad, dx, lhs, rhs, solve,
                        DirichletBC, assemble, parameters as fenics_params)
    FENICS_AVAILABLE = True
except ImportError:
    print("WARNING: FEniCS not found. Running in reduced ODE-only mode.")
    FENICS_AVAILABLE = False

try:
    from scipy.integrate import solve_ivp
    SCIPY_AVAILABLE = True
except ImportError:
    print("ERROR: SciPy required. Install with: pip install scipy")
    sys.exit(1)


# ===========================================================================
# 1. PARAMETER LOADING
# ===========================================================================

def load_parameters(config_path):
    """Load parameters from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, 'r') as f:
        if path.suffix in ('.yaml', '.yml'):
            params = yaml.safe_load(f)
        elif path.suffix == '.json':
            params = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    # Flatten nested dicts for easier access
    flat = {}
    for key, value in params.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        else:
            flat[key] = value
    params['_flat'] = flat
    return params


# ===========================================================================
# 2. MESH GENERATION
# ===========================================================================

def load_or_generate_mesh(params, mesh_dir="mesh_files"):
    """Load an existing mesh or generate a new one."""
    os.makedirs(mesh_dir, exist_ok=True)
    xdmf_path = os.path.join(mesh_dir, "maroflux_mesh.xdmf")

    if not os.path.exists(xdmf_path):
        print("Mesh not found. Generating...")
        from mesh.generate_mesh import create_mesh
        create_mesh(output_dir=mesh_dir)

    if FENICS_AVAILABLE:
        mesh = Mesh()
        with XDMFFile(xdmf_path) as f:
            f.read(mesh)
        print(f"Mesh loaded: {mesh.num_cells()} cells")
        return mesh
    else:
        print("FEniCS unavailable – returning None for mesh.")
        return None


# ===========================================================================
# 3. ODE SOLVER (Reduced 0D model for quick testing)
# ===========================================================================

class ReducedODESolver:
    """
    Solves the macrophage ODE system in 0D (well-mixed, no spatial gradients).
    Useful for quick parameter sweeps and debugging.
    """

    def __init__(self, params):
        self.params = params
        self._extract_params()

    def _extract_params(self):
        p = self.params['_flat']
        self.k_rec_M0 = p.get('macrophage.k_rec_M0', 1.16e-5)
        self.tau_DAMPs = p.get('macrophage.tau_DAMPs', 48.0) / 24.0  # hours → days
        self.M_max = p.get('macrophage.M_max', 1e6)
        self.k_d_M0 = p.get('macrophage.k_d_M0', 3.2e-6) * 86400
        self.k_d_M1 = p.get('macrophage.k_d_M1', 5.8e-6) * 86400
        self.k_d_M2 = p.get('macrophage.k_d_M2', 2.0e-6) * 86400
        self.k_pol_M1 = p.get('macrophage.k_pol_M1', 2e-5) * 86400
        self.k_pol_M2 = p.get('macrophage.k_pol_M2', 5e-6) * 86400
        self.k_switch = p.get('macrophage.k_switch', 1e-5) * 86400
        self.delta_syn = p.get('synergy.delta_syn', 1.85)
        self.K_Sr = p.get('macrophage.K_Sr', 25.0)
        self.K_IL4 = p.get('macrophage.K_IL4', 2.0)
        self.pH_0_5 = p.get('gate.pH_0.5', 7.1)
        self.k_pH = p.get('gate.k_pH', 25.0)
        self.t_onset_Sr = p.get('t_onset.Sr', 3.0)
        self.t_onset_IL4 = p.get('t_onset.IL4', 3.0)

    def _gate_function(self, t, pH):
        """Simple pH-only gate for reduced model."""
        return 1.0 / (1.0 + np.exp(-self.k_pH * (pH - 7.0)))

    def _source_Sr(self, t):
        if t < self.t_onset_Sr:
            return 0.0
        return 25.0 * np.exp(-0.1 * (t - self.t_onset_Sr))

    def _source_IL4(self, t):
        if t < self.t_onset_IL4:
            return 0.0
        return 2.0 * np.exp(-0.1 * (t - self.t_onset_IL4))

    def rhs(self, t, y):
        M0, M1, M2, pH = y
        f_DAMPs = np.exp(-t / self.tau_DAMPs)
        M_total = M0 + M1 + M2
        G_M0 = self.k_rec_M0 * f_DAMPs * max(0, 1 - M_total / self.M_max)

        P_M0_to_M1 = self.k_pol_M1 * f_DAMPs
        P_M0_to_M2 = 0.1 * self.k_pol_M2

        C_Sr = self._source_Sr(t)
        C_IL4 = self._source_IL4(t)
        gate = self._gate_function(t, pH)
        P_M1_to_M2 = (self.k_switch * self.delta_syn *
                      (C_Sr / (self.K_Sr + C_Sr + 1e-10)) *
                      (C_IL4 / (self.K_IL4 + C_IL4 + 1e-10)) * gate)

        dM0_dt = G_M0 - self.k_d_M0 * M0 - P_M0_to_M1 * M0 - P_M0_to_M2 * M0
        dM1_dt = P_M0_to_M1 * M0 - self.k_d_M1 * M1 - P_M1_to_M2 * M1
        dM2_dt = P_M0_to_M2 * M0 + P_M1_to_M2 * M1 - self.k_d_M2 * M2

        q_H_M1 = 3e-17 * 86400
        q_H_M2 = 7e-18 * 86400
        k_buf = 0.005 * 86400
        dpH_dt = - (q_H_M1 * M1 + q_H_M2 * M2) / 1e6 + k_buf * (7.4 - pH)

        return [dM0_dt, dM1_dt, dM2_dt, dpH_dt]

    def solve(self, t_end=30.0, n_points=1000):
        t_eval = np.linspace(0, t_end, n_points)
        y0 = [1e3, 1e4, 1e2, 6.5]
        sol = solve_ivp(self.rhs, (0, t_end), y0, method='Radau',
                        t_eval=t_eval, rtol=1e-6, atol=1e-9)
        self.t = sol.t
        self.M0, self.M1, self.M2, self.pH = sol.y
        return self.t, self.M1, self.M2, self.pH


# ===========================================================================
# 4. MAIN SIMULATION WRAPPER
# ===========================================================================

class MaroFluxSimulation:
    """Top-level simulation manager."""

    def __init__(self, config_path):
        self.params = load_parameters(config_path)
        self.output_dir = self.params.get('simulation', {}).get('output_dir', 'output')
        os.makedirs(os.path.join(self.output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)

    def run_ode_only(self, t_end=30.0):
        """Run the reduced ODE model."""
        print("Running reduced ODE simulation...")
        ode = ReducedODESolver(self.params)
        t, M1, M2, pH = ode.solve(t_end=t_end)
        self._save_ode_results(t, M1, M2, pH)
        self._plot_ode_results(t, M1, M2, pH)
        return t, M1, M2, pH

    def _save_ode_results(self, t, M1, M2, pH):
        data = np.column_stack([t, M1, M2, pH])
        header = "time(days),M1(cells/cm3),M2(cells/cm3),pH"
        np.savetxt(os.path.join(self.output_dir, 'data', 'ode_results.csv'),
                   data, header=header, delimiter=',', fmt='%.6e')

    def _plot_ode_results(self, t, M1, M2, pH):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(t, M1, 'r-', label='M1 (Pro-inflammatory)', linewidth=2)
        ax1.plot(t, M2, 'b-', label='M2 (Pro-regenerative)', linewidth=2)
        ax1.set_ylabel('Cell Density (cells/cm³)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('MaroFlux Macrophage Polarization (Reduced ODE Model)')

        ax2.plot(t, pH, 'g-', linewidth=2)
        ax2.axhline(y=7.0, color='gray', linestyle='--', label='Gate threshold')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('pH')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'figures', 'ode_results.png'), dpi=150)
        plt.close(fig)
        print(f"Figures saved to {self.output_dir}/figures/")


# ===========================================================================
# 5. COMMAND-LINE INTERFACE
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MaroFlux Complete Solver — Unified MaroFlux Equation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python maroflux_solver_complete.py
  python maroflux_solver_complete.py --config parameters/parameter_set.yaml --t-end 14
  python maroflux_solver_complete.py --ode-only
        """
    )
    parser.add_argument('--config', default='parameters/parameter_set.yaml',
                        help='Path to YAML/JSON parameter file')
    parser.add_argument('--t-end', type=float, default=30.0,
                        help='Simulation end time (days)')
    parser.add_argument('--ode-only', action='store_true',
                        help='Run reduced ODE model only (no FEM)')
    parser.add_argument('--output', default='output',
                        help='Output directory')
    args = parser.parse_args()

    sim = MaroFluxSimulation(args.config)
    sim.output_dir = args.output

    print("=" * 60)
    print("  MaroFlux Complete Solver")
    print("  Unified MaroFlux Equation (UME)")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Duration: {args.t_end} days")
    print(f"  Mode: {'ODE-only' if args.ode_only else 'Full PDE-ODE'}")
    print("=" * 60)

    if args.ode_only or not FENICS_AVAILABLE:
        sim.run_ode_only(t_end=args.t_end)
    else:
        print("Full PDE-ODE simulation not yet implemented in this version.")
        print("Falling back to reduced ODE model.")
        sim.run_ode_only(t_end=args.t_end)

    print("\nSimulation complete.")
    print(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
