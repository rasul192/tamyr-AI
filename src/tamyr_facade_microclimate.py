"""
Tamyr AI: 2.5D Facade Microclimate Discretization Engine.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from src.tamyr_physics_step import MaterialProfile, compute_daily_damage_step, PROFILE_YASAWI


def compute_facade_damage_distribution(
    climate_df: pd.DataFrame,
    material: MaterialProfile = PROFILE_YASAWI,
    dominant_wind_dir_deg: float = 45.0
) -> Dict[str, Any]:
    facades = {
        "South": {"azimuth": 180.0, "solar_gain": 1.45},
        "North": {"azimuth": 0.0,   "solar_gain": 0.90},
        "East":  {"azimuth": 90.0,  "solar_gain": 1.20},
        "West":  {"azimuth": 270.0, "solar_gain": 1.25},
    }

    results = {}
    n_days = len(climate_df)
    t_min_base = climate_df["t_min"].to_numpy()
    t_max_base = climate_df["t_max"].to_numpy()
    rh_base = climate_df["rh"].to_numpy()
    v_base = climate_df["v_max"].to_numpy()

    for f_name, f_params in facades.items():
        cumulative_d = 0.0
        stresses_sum = {"S_therm": 0.0, "S_cryo": 0.0, "S_salt": 0.0, "S_wind": 0.0, "S_seism": 0.0}

        for k in range(n_days):
            t_max_wall = t_max_base[k] + (f_params["solar_gain"] - 1.0) * 12.0
            t_min_wall = t_min_base[k]
            angle_diff = np.radians(f_params["azimuth"] - dominant_wind_dir_deg)
            v_eff = v_base[k] * max(0.15, np.cos(angle_diff))
            rh_wall = np.clip(rh_base[k] / (f_params["solar_gain"] ** 0.5), 10.0, 98.0)

            d_inc, stresses = compute_daily_damage_step(
                t_min=float(t_min_wall),
                t_max=float(t_max_wall),
                rh=float(rh_wall),
                v_max=float(v_eff),
                a_peak=0.02,
                material=material,
                dt=1.0
            )
            cumulative_d += d_inc
            for s_key in stresses_sum:
                stresses_sum[s_key] += stresses[s_key]

        results[f_name] = {
            "cumulative_delta_d": round(cumulative_d, 4),
            "leading_stress": max(stresses_sum, key=stresses_sum.get),
            "stress_sums": {k: round(v, 1) for k, v in stresses_sum.items()}
        }

    return results
