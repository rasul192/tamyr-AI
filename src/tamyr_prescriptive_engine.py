"""
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
