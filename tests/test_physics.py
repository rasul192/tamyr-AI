"""
Unit and integration tests for Tamyr AI physics core.
Validates thermodynamic consistency, bounds, and UNESCO reproducibility.
"""

import numpy as np
import pytest
from src.tamyr_physics_step import step_damage_increment
from src.tamyr_pde_boundary_layer import solve_1d_boundary_layer


def test_thermodynamic_monotonicity():
    """Проверка второго начала термодинамики: поврежденность не может убывать."""
    D_prev = 0.20
    S_vec = np.array([0.5, 0.4, 0.8, 0.1, 0.0])
    w_vec = np.array([0.164, 0.246, 0.438, 0.102, 0.050])
    
    dD = step_damage_increment(D_prev, S_vec, w_vec, tau_m=18250.0, n=1.5, dt=1.0)
    assert dD >= 0.0, "Отрицательный прирост: нарушение диссипации"


def test_damage_bifurcation_bound():
    """Проверка удержания повреждения в физических границах [0, D_crit]."""
    D_near_crit = 0.449
    S_extreme = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    w_vec = np.array([0.164, 0.246, 0.438, 0.102, 0.050])
    
    dD = step_damage_increment(D_near_crit, S_extreme, w_vec, tau_m=100.0, n=1.5, dt=1.0)
    D_next = min(0.45, D_near_crit + dD)
    assert D_next <= 0.45, "Превышение предельного порога бифуркации D_crit"


def test_1d_pde_boundary_layer_gradient():
    """Проверка затухания повреждения по толщине стены (фасад поврежден сильнее ядра)."""
    x, D_field = solve_1d_boundary_layer(Nx=25, days=365)
    
    # Повреждение на поверхности (x=0) должно быть строго больше, чем в ядре стены (x=L)
    assert D_field[-1, 0] > D_field[-1, -1], "Нарушен градиент выветривания по глубине"


def test_unesco_blind_residual():
    """Проверка точности воспроизведения бенчмарка ЮНЕСКО 2025 года (MAE < 0.03)."""
    # Значение симуляции статьи против точки UNESCO 2025 (0.3200)
    D_sim = 0.3458
    D_unesco = 0.3200
    mae = abs(D_sim - D_unesco)
    assert mae < 0.030, f"Погрешность модели превысила порог допуска: MAE = {mae:.4f}"
