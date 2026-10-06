"""
Tamyr AI: Physics-Informed Continuum Damage Step Engine (v2.0).
Computes daily damage increment dD using dimensionless dissipative potentials.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class PhysicsConstants:
    DELTA_T_CRIT: float = 30.0   # [°C]  Критический суточный градиент
    RH_CRIT: float = 80.0        # [%]   Порог капиллярного насыщения для льда
    V_THRESH: float = 7.0        # [м/с] Порог отрыва песка (закон Бэгнольда)
    V_REF: float = 20.0          # [м/с] Штормовая референсная скорость
    T_REF_SALT: float = 45.0     # [°C]  Базовая температура испарительного галогенеза
    A_CRIT: float = 0.40         # [g]   Критическое ускорение грунта (PGA)


@dataclass(frozen=True)
class MaterialProfile:
    name: str
    tau_m: float    # [сутки] Характерный ресурс несущей способности
    w_salt: float   # Доля солевой кристаллизации
    w_cryo: float   # Доля морозного пучения
    w_therm: float  # Доля термоупругой усталости
    w_wind: float   # Доля эоловой абразии
    w_seism: float  # Доля сейсмических деформаций

    def validate(self) -> None:
        total = self.w_salt + self.w_cryo + self.w_therm + self.w_wind + self.w_seism
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Сумма весов для {self.name} должна быть 1.0 (получено {total:.4f})")


PROFILE_YASAWI = MaterialProfile(
    name="Mausoleum_Yasawi_Fired_Brick",
    tau_m=18250.0,
    w_salt=0.40,
    w_cryo=0.25,
    w_therm=0.15,
    w_wind=0.10,
    w_seism=0.10,
)

PROFILE_OTRAR = MaterialProfile(
    name="Otrar_Unbaked_Adobe",
    tau_m=3650.0,
    w_salt=0.15,
    w_cryo=0.30,
    w_therm=0.10,
    w_wind=0.40,
    w_seism=0.05,
)

PROFILE_TAMGALY = MaterialProfile(
    name="Tamgaly_Metamorphic_Sandstone",
    tau_m=73000.0,
    w_salt=0.05,
    w_cryo=0.30,
    w_therm=0.40,
    w_wind=0.15,
    w_seism=0.10,
)


def compute_daily_damage_step(
    t_min: float,
    t_max: float,
    rh: float,
    v_max: float,
    a_peak: float = 0.02,
    material: MaterialProfile = PROFILE_YASAWI,
    const: PhysicsConstants = PhysicsConstants(),
    dt: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    delta_t = abs(t_max - t_min)
    s_therm = min(1.0, (delta_t / const.DELTA_T_CRIT) ** 2)

    if t_min <= 0.0 and rh > const.RH_CRIT:
        s_cryo = min(1.0, (rh - const.RH_CRIT) / (100.0 - const.RH_CRIT))
    else:
        s_cryo = 0.0

    if t_max > 5.0:
        s_salt = min(1.0, (t_max / const.T_REF_SALT) * (1.0 - (rh / 100.0)))
    else:
        s_salt = 0.0

    if v_max > const.V_THRESH:
        v_ratio = (v_max - const.V_THRESH) / (const.V_REF - const.V_THRESH)
        s_wind = min(1.0, v_ratio ** 3)
    else:
        s_wind = 0.0

    s_seism = min(1.0, a_peak / const.A_CRIT)

    weighted_stress = (
        material.w_therm * s_therm
        + material.w_cryo * s_cryo
        + material.w_salt * s_salt
        + material.w_wind * s_wind
        + material.w_seism * s_seism
    )

    delta_d = (weighted_stress / material.tau_m) * dt

    stresses = {
        "S_therm": round(s_therm, 6),
        "S_cryo": round(s_cryo, 6),
        "S_salt": round(s_salt, 6),
        "S_wind": round(s_wind, 6),
        "S_seism": round(s_seism, 6),
    }

    return delta_d, stresses
