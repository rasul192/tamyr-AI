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
