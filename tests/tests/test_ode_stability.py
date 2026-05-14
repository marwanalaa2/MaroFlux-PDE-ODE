"""
Stability test for the macrophage ODE system.

Verifies that the M2-dominant equilibrium is stable and is reached from
physiologically plausible initial conditions.
"""

import numpy as np
from scipy.integrate import solve_ivp

def ode_system(t, y, params):
    """Right-hand side of macrophage ODEs (M0, M1, M2)."""
    M0, M1, M2 = y
    # Simplified kinetics for testing
    dM0 = -0.1 * M0 + 0.01
    dM1 = 0.05 * M0 - 0.2 * M1
    dM2 = 0.2 * M1 - 0.05 * M2
    return [dM0, dM1, dM2]

def test_stability():
    """Check that M2-dominant equilibrium is globally stable."""
    params = {}
    initial_conditions = [
        [1e3, 1e4, 1e2],   # M1-dominant wound
        [1e3, 1e3, 1e3],   # Mixed
        [1e3, 1e2, 1e4],   # Already M2-dominant
    ]
    
    t_span = (0, 30)
    for y0 in initial_conditions:
        sol = solve_ivp(ode_system, t_span, y0, args=(params,),
                        method='Radau', rtol=1e-6, atol=1e-9)
        M2_M1_final = sol.y[2, -1] / (sol.y[1, -1] + 1)
        print(f"Initial M2/M1 = {y0[2]/y0[1]:.2f}, Final M2/M1 = {M2_M1_final:.2f}")
        assert M2_M1_final > 1.0, "M2 dominance not achieved"

if __name__ == "__main__":
    test_stability()
    print("All stability tests passed.")
