"""
Quick sanity check before commit: verifies modules and runs 1-year dry simulation.
"""

import numpy as np
import pandas as pd
from src.tamyr_physics_step import PROFILE_YASAWI
from src.tamyr_integrator import ContinuumDamageSolver


def main():
    print("--- Проверка ядра Tamyr AI v2.0 ---")
    PROFILE_YASAWI.validate()

    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    df_test = pd.DataFrame({
        "t_min": np.random.uniform(-10, 20, 365),
        "t_max": np.random.uniform(5, 40, 365),
        "rh": np.random.uniform(15, 90, 365),
        "v_max": np.random.uniform(2, 12, 365),
        "a_peak": np.full(365, 0.02),
    })

    solver = ContinuumDamageSolver(material=PROFILE_YASAWI)
    result = solver.solve(df_test, initial_damage=0.28)

    print("Статус: УСПЕШНО")
    print("Начальный D_0: 0.2800")
    print(f"Конечный D(365): {result.final_damage:.4f}")
    print(f"Прирост за год: +{result.final_damage - 0.28:.4f}")


if __name__ == "__main__":
    main()
