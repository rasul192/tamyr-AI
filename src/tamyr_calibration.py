"""
Tamyr AI: Constrained SLSQP Parameter Calibration against UNESCO SOC Benchmarks.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .tamyr_physics_step import MaterialProfile, PhysicsConstants
from .tamyr_integrator import ContinuumDamageSolver


class MaterialCalibrator:
    def __init__(
        self,
        climate_df: pd.DataFrame,
        ground_truth_points: List[Tuple[int, float, float]],
        prior_profile: MaterialProfile,
        regularization_gamma: float = 1e-3,
    ):
        self.climate_df = climate_df
        self.gt_points = ground_truth_points
        self.prior_profile = prior_profile
        self.gamma = regularization_gamma

        self.prior_vector = np.array([
            prior_profile.w_therm,
            prior_profile.w_cryo,
            prior_profile.w_salt,
            prior_profile.w_wind,
            prior_profile.w_seism,
            prior_profile.tau_m,
        ])

    def _objective(self, theta: np.ndarray, initial_d: float) -> float:
        w_th, w_cr, w_sa, w_wi, w_se, tau_m = theta

        profile = MaterialProfile(
            name="Optimizing_Profile",
            tau_m=tau_m,
            w_therm=w_th,
            w_cryo=w_cr,
            w_salt=w_sa,
            w_wind=w_wi,
            w_seism=w_se,
        )

        solver = ContinuumDamageSolver(material=profile)
        res = solver.solve(self.climate_df, initial_damage=initial_d)

        residual = 0.0
        for idx, d_real, sigma in self.gt_points:
            d_sim = res.damage_trajectory[min(idx, len(res.damage_trajectory) - 1)]
            residual += ((d_sim - d_real) / sigma) ** 2
        residual /= len(self.gt_points)

        theta_norm = theta.copy()
        theta_norm[5] /= 20000.0
        prior_norm = self.prior_vector.copy()
        prior_norm[5] /= 20000.0

        reg_loss = self.gamma * np.sum((theta_norm - prior_norm) ** 2)
        return float(residual + reg_loss)

    def fit(self, initial_d: float) -> MaterialProfile:
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x[:5]) - 1.0})
        bounds = [
            (0.05, 0.70),
            (0.05, 0.70),
            (0.05, 0.70),
            (0.05, 0.70),
            (0.02, 0.30),
            (10000.0, 35000.0),
        ]

        res = minimize(
            fun=self._objective,
            x0=self.prior_vector,
            args=(initial_d,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6, "disp": False},
        )

        best = res.x
        return MaterialProfile(
            name=f"{self.prior_profile.name}_Calibrated",
            tau_m=float(best[5]),
            w_therm=float(best[0]),
            w_cryo=float(best[1]),
            w_salt=float(best[2]),
            w_wind=float(best[3]),
            w_seism=float(best[4]),
        )
