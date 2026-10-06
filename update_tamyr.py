"""
Tamyr AI v2.0: Auto-updater for repository.
Adds prescriptive engine, facade microclimate, pipeline controller, and README.md,
then automatically commits and pushes everything to GitHub.
"""

from pathlib import Path
import subprocess

# Безопасное формирование кавычек для Markdown
FENCE = chr(96) * 3

FILES = {
    "src/tamyr_prescriptive_engine.py": '''"""
Tamyr AI: Prescriptive Conservation Decision Engine (v2.0).
Translates damage state vector [D, dD/dt] into prioritized heritage interventions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from src.tamyr_physics_step import MaterialProfile, PROFILE_YASAWI, PROFILE_OTRAR


class RiskLevel(Enum):
    GREEN = "Normal (Безопасный)"
    YELLOW = "Attention (Плановое обслуживание)"
    ORANGE = "Warning (Интенсивная деградация)"
    RED = "Critical (Аварийная опасность)"


@dataclass
class ConservationAction:
    risk_level: RiskLevel
    title: str
    urgency_days: int
    recommended_methods: List[str]
    estimated_life_extension_years: float


class PrescriptiveEngine:
    def __init__(self, d_attention: float = 0.25, d_critical: float = 0.45, v_threshold: float = 0.04):
        self.d_att = d_attention
        self.d_crit = d_critical
        self.v_th = v_threshold

    def evaluate_state(self, current_d: float, velocity_annual: float) -> RiskLevel:
        if current_d >= self.d_crit or velocity_annual >= 2 * self.v_th:
            return RiskLevel.RED
        elif current_d >= self.d_att and velocity_annual < self.v_th:
            return RiskLevel.YELLOW
        elif current_d < self.d_att and velocity_annual >= self.v_th:
            return RiskLevel.ORANGE
        else:
            return RiskLevel.GREEN

    def generate_recommendation(
        self,
        material: MaterialProfile,
        current_d: float,
        velocity_annual: float,
        stress_shares: Dict[str, float]
    ) -> ConservationAction:
        level = self.evaluate_state(current_d, velocity_annual)
        top_driver = max(stress_shares, key=stress_shares.get) if stress_shares else "S_salt"

        if level == RiskLevel.RED:
            return ConservationAction(
                risk_level=level,
                title="Экстренная конструктивная стабилизация и снятие напряжений",
                urgency_days=30,
                recommended_methods=[
                    "Глубинная инъекция микроэмульсий гидравлической извести",
                    "Установка временных разгрузочных кружал",
                    "Лазерное 3D-сканирование деформаций сводов"
                ],
                estimated_life_extension_years=15.0
            )
        elif level == RiskLevel.ORANGE:
            methods = [
                "Обустройство пристенного вентилируемого дренажа",
                "Наложение обессоливающих компрессов из сепиолита",
                "Прекращение ирригации в буферной зоне"
            ] if "salt" in top_driver else [
                "Установка ветрозащитных экранов",
                "Нанесение гидрофобных паропроницаемых пропиток"
            ]
            return ConservationAction(
                risk_level=level,
                title=f"Купирование активного климатического стресса ({top_driver})",
                urgency_days=90,
                recommended_methods=methods,
                estimated_life_extension_years=8.0
            )
        elif level == RiskLevel.YELLOW:
            return ConservationAction(
                risk_level=level,
                title="Планово-профилактическая консервация защитного слоя",
                urgency_days=180,
                recommended_methods=[
                    "Расчистка и перерасшивка выветрившихся швов известковым раствором",
                    "Биоцидная обработка очагов лишайников",
                    "Локальная докомпоновка утраченных фрагментов"
                ],
                estimated_life_extension_years=5.0
            )
        else:
            return ConservationAction(
                risk_level=level,
                title="Регламентный эксплуатационный мониторинг",
                urgency_days=365,
                recommended_methods=[
                    "Визуальная фотофиксация контрольных реперов",
                    "Проверка работоспособности системы водоотведения"
                ],
                estimated_life_extension_years=0.0
            )
''',

    "src/tamyr_facade_microclimate.py": '''"""
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
''',

    "src/tamyr_pipeline_integration.py": '''"""
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
''',

    "README.md": f"""# Tamyr AI: Physics-Informed Predictive Preservation Engine (v2.0)

Платформа предиктивного анализа и мониторинга объектов культурного наследия Южного Казахстана на основе механики сплошных сред (Continuum Damage Mechanics).

## Быстрый старт для напарника

### 1. Установка зависимостей
{FENCE}bash
pip install -r requirements.txt
{FENCE}

### 2. Запуск проверки ядра
{FENCE}bash
python run_verification.py
{FENCE}

### 3. Как использовать в Streamlit (app.py)
{FENCE}python
import pandas as pd
from src.tamyr_pipeline_integration import TamyrPipeline

df = pd.read_csv("data/turkestan_climate.csv")

pipeline = TamyrPipeline()
result = pipeline.run(df, site_key="yasawi", initial_d=0.28)

print("Статус:", result.risk_status)
print("Итоговый износ D:", result.final_damage)
print("Рекомендации:", result.recommended_methods)
print("Вклад стрессов (%):", result.stress_shares_percent)
print("Фасады:", result.facade_anisotropy)
{FENCE}

## Структура модулей (src/)
- `tamyr_physics_step.py` — расчет суточного приращения повреждения по 5 стресс-потенциалам.
- `tamyr_integrator.py` — численный интегратор траектории D(t).
- `tamyr_prescriptive_engine.py` — классификация рисков и рекомендации по регламентам ICOMOS.
- `tamyr_facade_microclimate.py` — расчет микроклимата и износа по 4 сторонам света.
- `tamyr_pipeline_integration.py` — мастер-контроллер конвейера для интеграции с дашбордом.
"""
}


def main():
    print("=== ОБНОВЛЕНИЕ РЕПОЗИТОРИЯ ДЛЯ НАПАРНИКА ===")
    for path_str, content in FILES.items():
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print(f"[+] Записан файл: {path_str}")

    print("\n--- Фиксация изменений в Git и отправка на GitHub ---")
    subprocess.run("git add .", shell=True, check=True)
    subprocess.run('git commit -m "feat: add prescriptive engine, facade 2.5D, and master pipeline"', shell=True, check=True)
    subprocess.run("git push origin main", shell=True, check=True)

    try:
        Path(__file__).unlink()
        print("[+] Скрипт обновления update_tamyr.py удален.")
    except Exception:
        pass

    print("\nГОТОВО! Все файлы загружены на GitHub.")


if __name__ == "__main__":
    main()