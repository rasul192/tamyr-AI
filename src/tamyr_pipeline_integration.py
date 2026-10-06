"""
Tamyr AI: Master Pipeline Controller.
Unified entry point for Streamlit UI and Analytics.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import time
import numpy as np
import pandas as pd

from src.tamyr_physics_step import PROFILE_YASAWI, PROFILE_OTRAR, PROFILE_TAMGALY
from src.tamyr_integrator import ContinuumDamageSolver
from src.tamyr_facade_microclimate import compute_facade_damage_distribution
from src.tamyr_prescriptive_engine import PrescriptiveEngine


@dataclass
class PipelineOutput:
    site_name: str
    material_name: str
    initial_damage: float
    final_damage: float
    delta_damage: float
    mean_annual_velocity: float
    risk_status: str
    urgent_action: str
    recommended_methods: list
    facade_anisotropy: Dict[str, Any]
    stress_shares_percent: Dict[str, float]
    execution_time_ms: float


class TamyrPipeline:
    REGISTRY = {
        "yasawi": PROFILE_YASAWI,
        "otrar": PROFILE_OTRAR,
        "tamgaly": PROFILE_TAMGALY,
    }

    def __init__(self):
        self.prescriptive = PrescriptiveEngine()

    def run(self, climate_df: pd.DataFrame, site_key: str = "yasawi", initial_d: float = 0.18) -> PipelineOutput:
        start = time.perf_counter()
        material = self.REGISTRY.get(site_key.lower(), PROFILE_YASAWI)

        df = climate_df.copy()
        df["t_min"] = np.clip(df["t_min"].interpolate(), -50.0, 50.0)
        df["t_max"] = np.clip(df["t_max"].interpolate(), -40.0, 60.0)
        df["rh"] = np.clip(df["rh"].interpolate(), 1.0, 100.0)
        df["v_max"] = np.clip(df["v_max"].interpolate(), 0.0, 45.0)
        if "a_peak" not in df.columns:
            df["a_peak"] = 0.02

        solver = ContinuumDamageSolver(material=material)
        res = solver.solve(df, initial_damage=initial_d)

        facades = compute_facade_damage_distribution(df, material=material)

        cum = res.cumulative_stresses
        total_energy = (
            material.w_salt * np.sum(cum["S_salt"]) +
            material.w_cryo * np.sum(cum["S_cryo"]) +
            material.w_therm * np.sum(cum["S_therm"]) +
            material.w_wind * np.sum(cum["S_wind"]) +
            material.w_seism * np.sum(cum["S_seism"])
        ) or 1.0

        shares = {
            "salt": round(float(material.w_salt * np.sum(cum["S_salt"]) / total_energy * 100), 1),
            "cryo": round(float(material.w_cryo * np.sum(cum["S_cryo"]) / total_energy * 100), 1),
            "therm": round(float(material.w_therm * np.sum(cum["S_therm"]) / total_energy * 100), 1),
            "wind": round(float(material.w_wind * np.sum(cum["S_wind"]) / total_energy * 100), 1),
            "seism": round(float(material.w_seism * np.sum(cum["S_seism"]) / total_energy * 100), 1),
        }

        ann_v = float(np.mean(res.damage_velocity))
        rec = self.prescriptive.generate_recommendation(
            material=material,
            current_d=res.final_damage,
            velocity_annual=ann_v,
            stress_shares={f"S_{k}": v / 100.0 for k, v in shares.items()}
        )

        elapsed = (time.perf_counter() - start) * 1000.0

        return PipelineOutput(
            site_name=site_key.upper(),
            material_name=material.name,
            initial_damage=round(initial_d, 4),
            final_damage=round(res.final_damage, 4),
            delta_damage=round(res.final_damage - initial_d, 4),
            mean_annual_velocity=round(ann_v, 5),
            risk_status=rec.risk_level.value,
            urgent_action=rec.title,
            recommended_methods=rec.recommended_methods,
            facade_anisotropy=facades,
            stress_shares_percent=shares,
            execution_time_ms=round(elapsed, 2)
        )
