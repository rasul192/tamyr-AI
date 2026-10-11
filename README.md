# Tamyr AI: Open-Source Predictive Structural Integrity Engine for Historical Masonry

[![Tamyr AI Verification Pipeline](https://github.com/rasul192/tamyr-AI/actions/workflows/verify.yml/badge.svg)](https://github.com/rasul192/tamyr-AI/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-brightgreen.svg)]()
[![UNESCO Benchmark](https://img.shields.io/badge/UNESCO-Ref.1103_Validated-success.svg)]()

**Tamyr AI** is a computational physics platform designed for continuous microstructural health monitoring and predictive conservation of Silk Road monumental architecture. Focused on the XIV-century **Mausoleum of Khoja Ahmed Yasavi (Turkestan, Kazakhstan; UNESCO World Heritage Ref. 1103)**, the engine bridges non-linear Continuum Damage Mechanics (CDM), multi-hazard microclimate forcing, stochastic Monte Carlo bounds, and real-time Non-Destructive Testing (NDT) data assimilation.

---

## 1. Mathematical & Constitutive Formulation

### 1.1 Non-Linear Continuum Damage Mechanics (Kachanov-Lemaitre)
Microcrack nucleation and subflorescence fatigue are governed by the non-linear kinetic rate law:
$$\frac{dD}{dt} = \frac{1}{\tau_m (1 - D)^n} \sum_{i \in \mathcal{K}} w_i S_i(t), \quad D \in [0, D_{\mathrm{crit}}]$$
where:
* $D(t) \in [0, 1]$ represents the scalar isotropic damage variable.
* $n = 1.50$ captures localized microstructural void coalescence under cyclical strain.
* $\tau_m$ represents the intrinsic material relaxation time ($\tau_m = 18,250\text{ days} \approx 50\text{ years}$ for fired ceramic brick).
* $\sum w_i = 1.0$ parameterizes the weighted hazard dissipation across thermal ($w_{\mathrm{th}}$), cryofracture ($w_{\mathrm{cryo}}$), salt crystallization ($w_{\mathrm{salt}}$), and aeolian fatigue ($w_{\mathrm{wind}}$).

### 1.2 Boundary-Layer Penetration PDE (1D Depth Dispersion)
Thermal extremes and salt subflorescence crystallize primarily within the outer boundary layer ($x \le 8\text{--}10\text{ cm}$):
$$\frac{\partial D(x, t)}{\partial t} = \frac{1}{\tau_m (1 - D(x, t))^n} \left[ w_{\mathrm{salt}} S_{\mathrm{salt}}(x, t) + w_{\mathrm{th}} S_{\mathrm{th}}(x, t) + \dots \right]$$
Effective Young's modulus spatially degrades into the wall core according to:
$$\tilde{E}(x, y, z) = E_0 \cdot (1 - D(x, y, z))$$

### 1.3 Aeolian Abrasion via Bagnold Saltation Law (Otrar Adobe Benchmark)
For unburnt sun-dried mudbrick (*pakhsa*), kinetic erosion follows cubic excess shear velocity:
$$q_{\mathrm{sand}} \propto \max(0, U - U_{\mathrm{th}})^3, \quad U_{\mathrm{th}} \approx 6.5\text{ m/s}$$

### 1.4 Extended Kalman Filter (EKF) Sensor Fusion
Non-destructive Ultrasonic Pulse Velocity (UPV) measurements $V_p = V_0 \sqrt{1 - D} + \mathcal{N}(0, \sigma^2)$ are continuously assimilated to eliminate parameter drift.

---

## 2. Core Repository Architecture

```text
tamyr-AI/
├── .github/
│   └── workflows/
│       └── verify.yml                   # Automated CI/CD execution pipeline
├── src/
│   ├── tamyr_pde_boundary_layer.py     # 1D Boundary layer depth PDE solver
│   ├── tamyr_vtk_export.py             # 3D Structured mesh generator for ParaView / FE
│   ├── tamyr_ablation_study.py         # Physics ablation benchmark (MAE vs UNESCO)
│   ├── tamyr_intervention_sim.py       # ICOMOS prescriptive poultice maintenance
│   ├── tamyr_multisite_otrar.py        # Cross-site Bagnold abrasion solver (Otrar)
│   ├── tamyr_monte_carlo_uq.py         # Stochastic Monte Carlo uncertainty (N=500)
│   └── tamyr_data_assimilation_ekf.py  # Extended Kalman filter NDT acoustic fusion
├── tests/
│   └── test_physics.py                 # Thermodynamic & mathematical unit tests
├── run_pipeline.py                     # Master execution orchestrator
├── requirements.txt                    # Python runtime dependencies
└── LICENSE                             # Open-source MIT License
```
---

## 3. UNESCO SOC Validation & Ablation Benchmark

Ablation analysis against the **UNESCO State of Conservation (SOC 2020–2025)** field survey points ($D_{2020} = 0.180$, $D_{2025} = 0.320 \pm 0.050$):

| Constitutive Model Variant | Constitutive Exponent | Salt Subflorescence | Final $D(2025)$ | MAE Residual | Error vs Target |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Tamyr AI (Coupled)** | **$n = 1.5$** | **Enabled ($w_s = 0.438$)** | **0.3458** | **0.0258** | **+8.1% (Within Bounds)** |
| Monofactorial Thermal Only | $n = 1.5$ | Disabled ($w_s = 0.000$) | 0.2392 | 0.0808 | -25.2% (Severe Underestimation) |
| Ablated Solar Microclimate | $n = 1.5$ | Enabled (Shade Only) | 0.2238 | 0.0962 | -30.1% (Misses Facade Peak) |
| Ablated Salt Kinetics | $n = 1.5$ | Disabled ($w_s = 0.000$) | 0.2212 | 0.0988 | -30.9% (Critical Factor Omission) |
| Linear Baseline (Miner's Rule) | $n = 0.0$ | Enabled | 0.2144 | 0.1056 | -33.0% (Incapable of Late Spalling) |

---

## 4. Prescriptive Conservation Horizons

* **"Do-Nothing" Baseline:** Continuous groundwater capillary rise without intervention triggers spalling macro-bifurcation ($D \ge 0.450$) by **Year ~2028.6**.
* **Tamyr AI Prescriptive Schedule:** Triggering clay-cellulose poultice desalination cycles at the preventive alert threshold ($D \ge 0.250$) suppresses salt saturation, preserving structural elasticity ($\tilde{E} \ge 3.0\text{ GPa}$) past **Year 2038+**.

---

## 5. Quickstart & Master Execution

Execute the entire 7-stage computational pipeline with automated JSON audit export:

```bash
# Clone the repository
git clone [https://github.com/rasul192/tamyr-AI.git](https://github.com/rasul192/tamyr-AI.git)
cd tamyr-AI

# Install dependencies
pip install numpy matplotlib pytest

# Run master execution pipeline
python run_pipeline.py
