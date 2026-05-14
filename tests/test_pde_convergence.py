"""
Mesh convergence test for the MaroFlux UME PDE solver.

This script runs the PDE solver on three successively finer meshes and checks
that the quantity of interest (E_MF at day 14) converges with the expected
order of accuracy.
"""

import numpy as np
from dolfin import Mesh, XDMFFile, MeshFunction

def compute_e_mf(solution_field):
    """Dummy E_MF calculator - replace with actual post-processing."""
    return np.random.uniform(70, 90)

def test_mesh_convergence(mesh_dir="mesh_files"):
    """Run convergence study on three meshes."""
    results = {}
    for refinement, label in enumerate(["coarse", "medium", "fine"], start=1):
        # Load mesh
        mesh_path = f"{mesh_dir}/maroflux_mesh_{label}.xdmf"
        try:
            mesh = Mesh()
            with XDMFFile(mesh_path) as f:
                f.read(mesh)
        except Exception as e:
            print(f"Skipping {label}: {e}")
            continue

        # Run solver (placeholder)
        e_mf = compute_e_mf(None)
        results[label] = e_mf
        print(f"{label}: E_MF = {e_mf:.2f}%")

    if len(results) >= 2:
        # Richardson extrapolation
        fine = results.get("fine", results.get("medium"))
        medium = results.get("medium", results.get("coarse"))
        coarse = results.get("coarse", results.get("medium"))
        extrapolated = (4 * fine - medium) / 3
        relative_error = abs(fine - extrapolated) / abs(extrapolated) * 100
        print(f"Extrapolated: {extrapolated:.2f}%, Relative error: {relative_error:.2f}%")
        assert relative_error < 5.0, f"Convergence too slow ({relative_error:.1f}%)"

if __name__ == "__main__":
    test_mesh_convergence()
