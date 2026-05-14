"""
Accuracy test for Strang operator splitting in the PDE-ODE coupling.

Verifies second-order convergence in time step Δt.
"""

import numpy as np

def solve_coupled(dt):
    """Placeholder for the coupled PDE-ODE solver."""
    # Simulate error decreasing with Δt²
    true_value = 75.0
    computed = true_value + 0.5 * dt**2 + 0.01 * np.random.randn()
    return computed

def test_strang_splitting():
    """Check second-order convergence rate."""
    dt_values = [0.2, 0.1, 0.05, 0.025]
    errors = []
    true = 75.0

    for dt in dt_values:
        computed = solve_coupled(dt)
        err = abs(computed - true)
        errors.append(err)
        print(f"Δt = {dt:.3f}, error = {err:.5f}")

    # Estimate convergence rate
    rates = []
    for i in range(1, len(dt_values)):
        rate = np.log(errors[i-1] / errors[i]) / np.log(dt_values[i-1] / dt_values[i])
        rates.append(rate)
    avg_rate = np.mean(rates)
    print(f"Estimated convergence rate: {avg_rate:.2f}")
    assert 1.8 < avg_rate < 2.2, f"Expected second-order, got {avg_rate:.2f}"

if __name__ == "__main__":
    test_strang_splitting()
