# MaroFlux PDE-ODE Hybrid Simulation Suite

**A Unified Mathematical and Geochemical Framework for Temporally-Programmed Immunomodulatory Bioceramics and Neuromuscular Regeneration**

> **Author:** Marwan Alaa Mohamed Mohamed Raslan  
> **Affiliation:** Independent Researcher, Cairo, Egypt  
> **Manuscript:** *MaroFlux_Final_v6.pdf* — Theoretical & Computational Framework  
> **Article Type:** Comprehensive Theory and Perspective  
> **License:** MIT  
> **Zenodo DOI:** `10.5281/zenodo.[XXXXXX]` ← *replace before submission*

---

## 📋 Overview

This repository contains the complete computational reproducibility package for the **MaroFlux Paradigm** manuscript. All simulation code, calibrated parameters, and sensitivity results are provided to enable full independent verification of the theoretical claims.

The codebase implements:
- **PDE solver** (Implicit Backward Euler, Sec 3.7.1) for spatiotemporal ion transport
- **ODE solver** (LSODA, Sec 3.2) for macrophage polarization and muscle dynamics
- **Strang splitting** coupling scheme with O(Δt²) verified accuracy (Sec 3.7.3)
- **Bifurcation analysis** (Sec 3.3.3) — saddle-node threshold for k_gate
- **Sobol sensitivity analysis** (Sec 3.4.2) via Saltelli sampling
- **Monte Carlo UQ** (Sec 3.4.3) — 95% confidence intervals on all key outputs
- **6-panel publication figure** matching manuscript methodology

---

## 🗂️ Repository Structure

```
MaroFlux-PDE-ODE/
│
├── maroflux_solver_complete.py      # Main simulation code (all sections)
├── calibrated_params_pde_ode.json   # Calibrated parameters + results summary
├── sobol_sensitivity_results_pde_ode.json  # Sobol S1/ST indices
│
├── figures/
│   └── MaroFlux_Complete_Output.pdf # 6-panel figure (300 dpi, auto-generated)
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── LICENSE                          # MIT License
└── .gitignore                       # Git ignore rules
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- pip package manager

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/[author]/MaroFlux-PDE-ODE.git
cd MaroFlux-PDE-ODE

# Install dependencies
pip install -r requirements.txt

# Run the full simulation
python maroflux_solver_complete.py
```

### For Publication-Grade Results

Before final submission, increase the sampling for Sobol and Monte Carlo:

```python
# In maroflux_solver_complete.py, change:
sobol_res, problem = run_sobol_analysis(params, N_base=2**13)   # ~65,536 runs
mc_results = run_monte_carlo_uq(params, N_mc=10000)              # 10,000 ensemble runs
```

Estimated runtime on standard hardware (16-core, 64 GB RAM):
- Standard run (N_base=64, N_mc=500): ~2 minutes
- Publication run (N_base=8192, N_mc=10000): ~6–12 hours

---

## 🚀 Running the Simulation

```bash
python maroflux_solver_complete.py
```

### Expected Output

```
✅ Main simulation completed in 0.52s  (300 steps, Δt=0.1 day)
📉 Computing bifurcation diagram (30 k_gate values)...
   → Critical k_gate* ≈ 0.100 day⁻¹ (saddle-node threshold)
🎲 Monte Carlo UQ: N=500 runs...
   M2 ratio 95% CI: [0.61, 0.91]
   Robustness (M2>0.60): 94.7%
✅ Figure saved → figures/MaroFlux_Complete_Output.pdf
✅ Saved: calibrated_params_pde_ode.json
✅ Saved: sobol_sensitivity_results_pde_ode.json
```

### Generated Files

| File | Description |
|------|-------------|
| `figures/MaroFlux_Complete_Output.pdf` | 6-panel figure (A–F), 300 dpi |
| `calibrated_params_pde_ode.json` | All parameters, results, and Monte Carlo CI |
| `sobol_sensitivity_results_pde_ode.json` | Sobol S1, ST indices per parameter |

---

## 📊 Figure Panels

| Panel | Content | Manuscript Section |
|-------|---------|-------------------|
| **(A)** | Spatiotemporal ion release & pH evolution | Sec 3.1.1–3.1.2, Eq. 1–2 |
| **(B)** | M1→M2 macrophage polarization dynamics | Sec 3.2, Eqs 4–5 |
| **(C)** | Muscle CSA recovery + ±15% UQ band | Sec 4.3.2, Eqs 8–10 |
| **(D)** | Bifurcation diagram: k_gate vs M2 ratio | Sec 3.3.3, Fig 3.3.3.A |
| **(E)** | MaroFlux Number Ma evolution | Sec 9.2.1 |
| **(F)** | Sobol sensitivity indices (or E_MF fallback) | Sec 3.4.2 / 9.1.1 |

---

## 🔢 Key Calibrated Parameters

| Parameter | Value | Primary Reference | Tolerance |
|-----------|-------|-------------------|-----------|
| D_Si (ionic diffusivity) | 1.2×10⁻¹⁰ m²/s | Kandori et al. (2020) [Ref 42] | ±15% |
| k_gate (polymer permeability) | 0.12 day⁻¹ | Bertani et al. (2017) [Ref 25] | ±25% |
| δ_syn (ionic synergy) | 1.8 | Hoppe et al. (2013) [Ref 46] | ±20% |
| pH_threshold | 7.1 | Patel et al. (2021) [Ref 44] | ±0.1 pH |
| k_gate* (bifurcation) | ≈ 0.10 day⁻¹ | Computed (Sec 3.3.3) | — |

Full parameter table: see `calibrated_params_pde_ode.json` and **Table 3.4.1** in the manuscript.

---

## 🧮 Mathematical Framework

The simulation implements the **Unified MaroFlux Equation (UME)**:

```
∂C_i/∂t = D_i∇²C_i − v·∇C_i + R_i(C_i, pH, cells) + S_i(x,t)    [Eq. 1]
∂pH/∂t  = D_pH∇²pH − v·∇pH − k_norm(pH − pH_eq) + Σ αⱼRⱼ         [Eq. 2]
v        = −K∇P/μ                                                    [Eq. 3, Darcy]
d[M1]/dt = G_M1 − D_M1[M1] − T_{M1→M2}[M1]                         [Eq. 4]
d[M2]/dt = G_M2 − D_M2[M2] + T_{M1→M2}[M1]                         [Eq. 5]
d(CSA)/dt = k_growth·MPS − k_atrophy·MPB                            [Eq. 8]
```

**Numerical scheme:** Strang operator splitting (O(Δt²), Sec 3.7.3):
1. PDE half-step → 2. ODE full-step → 3. PDE half-step

**Stability:** Proven via Lax–Milgram theorem (existence/uniqueness) and Lyapunov energy method (global asymptotic stability to M2-dominant equilibrium). See **Sec 3.3.1–3.3.2**.

---

## 📌 Sections Implemented

| Code Function | Manuscript Section |
|--------------|-------------------|
| `MaroFluxParams` | Sec 3.4.1, Table 3.4.1 |
| `solve_pde_implicit` | Sec 3.7.1, Eq. 1 (backward Euler FDM) |
| `update_pH` | Sec 3.1.2, Eq. 2 |
| `ode_rhs` | Sec 3.2, 4.3.2, Eqs 4–5, 8–10 |
| `run_maroflux_simulation` | Sec 3.7.3 (Strang splitting algorithm) |
| `compute_bifurcation` | Sec 3.3.3 (saddle-node analysis) |
| `run_sobol_analysis` | Sec 3.4.2 (Saltelli + Sobol indices) |
| `run_monte_carlo_uq` | Sec 3.4.3 (Monte Carlo ensemble) |
| `generate_figures` | Panels A–F (Fig in manuscript) |

---

## ⚠️ Important Notes Before Submission

1. **Replace repository URL:** Change `[author]` to your actual GitHub username in all files.
2. **Replace Zenodo DOI:** After upload to Zenodo, replace `[XXXXXX]` with actual DOI.
3. **Publication N values:** Increase `N_base=2**13` and `N_mc=10000` before final run.
4. **SALib installation:** Required for Sobol panel in Figure F. Without it, the panel shows E_MF index instead.

---

## 📚 How to Cite

If you use this code in your research, please cite both the manuscript and this repository:

**Manuscript:**
> Raslan, M.A.M.M. (2026). *The MaroFlux Paradigm: A Unified Mathematical and Geochemical Framework for Temporally-Programmed Immunomodulatory Bioceramics and Neuromuscular Regeneration.* [Journal Name]. DOI: [manuscript DOI]

**Code Repository:**
> Raslan, M.A.M.M. (2026). *MaroFlux PDE-ODE Hybrid Simulation Suite* (v6). Zenodo. DOI: 10.5281/zenodo.[XXXXXX]

**BibTeX:**
```bibtex
@software{raslan2026maroflux,
  author    = {Raslan, Marwan Alaa Mohamed Mohamed},
  title     = {{MaroFlux PDE-ODE Hybrid Simulation Suite}},
  year      = {2026},
  version   = {v6},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.[XXXXXX]},
  url       = {https://github.com/[author]/MaroFlux-PDE-ODE}
}
```

---

## 🤝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.  
Reviewers and researchers are explicitly granted the right to run, inspect, and reuse this code for scientific purposes.

---

## 📬 Contact

**Marwan Alaa Mohamed Mohamed Raslan**  
Independent Researcher, Cairo, Egypt  
*Correspondence regarding this manuscript should be directed to the corresponding author.*
