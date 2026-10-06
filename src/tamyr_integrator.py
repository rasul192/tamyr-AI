"""
Tamyr AI: Numerical Integration Solver for Continuum Damage Trajectories.
Solves Cauchy problem on discrete daily meteorological arrays.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
from .tamyr_physics_step import MaterialProfile, PhysicsConstants, compute_daily_damage_step, PROFILE_YASAWI


@dataclass
class SimulationResult:
    timeline_days: np.ndarray
    damage_trajectory: np.ndarray
    damage_velocity: np.ndarray
    cumulative_stresses: Dict[str, np.ndarray]
    final_damage: float


class ContinuumDamageSolver:
    def __init__(
        self,
        material: MaterialProfile = PROFILE_YASAWI,
        constants: PhysicsConstants = PhysicsConstants(),
        dt: float = 1.0,
    ):
        self.material = material
        self.constants = constants
        self.dt = dt
        self.material.validate()

    def solve(self, climate_df: pd.DataFrame, initial_damage: float = 0.15) -> SimulationResult:
        n_steps = len(climate_df)
        t_days = np.arange(n_steps) * self.dt
        d_trajectory = np.zeros(n_steps, dtype=np.float64)
        d_velocity = np.zeros(n_steps, dtype=np.float64)

        stress_history = {
            "S_therm": np.zeros(n_steps, dtype=np.float64),
            "S_cryo": np.zeros(n_steps, dtype=np.float64),
            "S_salt": np.zeros(n_steps, dtype=np.float64),
            "S_wind": np.zeros(n_steps, dtype=np.float64),
            "S_seism": np.zeros(n_steps, dtype=np.float64),
        }

        current_d = float(initial_damage)
        d_trajectory[0] = current_d

        t_min_arr = climate_df["t_min"].to_numpy()
        t_max_arr = climate_df["t_max"].to_numpy()
        rh_arr = climate_df["rh"].to_numpy()
        v_max_arr = climate_df["v_max"].to_numpy()
        a_peak_arr = climate_df["a_peak"].to_numpy() if "a_peak" in climate_df.columns else np.full(n_steps, 0.02)

        for k in range(n_steps):
            d_inc, stresses = compute_daily_damage_step(
                t_min=float(t_min_arr[k]),
                t_max=float(t_max_arr[k]),
                rh=float(rh_arr[k]),
                v_max=float(v_max_arr[k]),
                a_peak=float(a_peak_arr[k]),
                material=self.material,
                const=self.constants,
                dt=self.dt,
            )

            for key in stress_history:
                stress_history[key][k] = stresses[key]

            d_velocity[k] = (d_inc / self.dt) * 365.25
            current_d = min(1.0, current_d + d_inc)

            if k + 1 < n_steps:
                d_trajectory[k + 1] = current_d

        return SimulationResult(
            timeline_days=t_days,
            damage_trajectory=d_trajectory,
            damage_velocity=d_velocity,
            cumulative_stresses=stress_history,
            final_damage=float(current_d),
        )
