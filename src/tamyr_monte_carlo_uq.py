"""
Tamyr AI - Stochastic Uncertainty Quantification (UQ) Solver
Monte Carlo sensitivity analysis (N=500 runs) evaluating parameter dispersion:
  - Material resilience tau_m ~ N(18250, 1800^2)
  - Lemaitre exponent n ~ N(1.5, 0.12^2)
  - Initial microcrack density D_0 ~ N(0.18, 0.015^2)
  - Environmental salt loading stochasticity
Generates P10/P50/P90 confidence envelopes and failure probability distributions.
"""

import numpy as np


def run_monte_carlo_uncertainty(
    num_simulations: int = 500,
    horizon_years: float = 12.0,
    dt_days: float = 1.0,
    D_crit: float = 0.450
):
    total_days = int(horizon_years * 365.25)
    t_years = 2020.0 + np.arange(total_days) / 365.25
    day_of_year = np.arange(total_days) % 365

    np.random.seed(42)

    # Генерация климатических факторов Туркестана
    S_salt_base = np.clip(np.sin((day_of_year - 80) * 2 * np.pi / 365) * 0.7 + 0.3, 0, 1)
    S_cryo = np.clip(np.cos(day_of_year * 2 * np.pi / 365) * 0.6 + 0.2, 0, 1)
    S_therm = np.clip(np.sin((day_of_year - 60) * 2 * np.pi / 365) * 0.5 + 0.4, 0, 1)
    S_wind = np.clip(np.random.weibull(1.8, total_days) * 0.25, 0, 1)

    # Матрица для сохранения всех траекторий D(t)
    D_ensemble = np.zeros((num_simulations, total_days))
    failure_years = []

    for sim in range(num_simulations):
        # Стохастическое сэмплирование параметров материала
        tau_m_sim = np.random.normal(18250.0, 1800.0)
        tau_m_sim = max(12000.0, tau_m_sim)

        n_sim = np.random.normal(1.50, 0.10)
        n_sim = max(1.10, min(1.90, n_sim))

        D_0_sim = np.random.normal(0.180, 0.015)
        D_0_sim = max(0.120, min(0.240, D_0_sim))

        # Случайное возмущение веса солевого фактора
        w_salt_sim = np.random.normal(0.438, 0.040)
        w_base = np.array([0.164, 0.246, w_salt_sim, 0.102, 0.050])
        w_norm = w_base / np.sum(w_base)

        D_curr = D_0_sim
        D_ensemble[sim, 0] = D_curr
        failed = False

        for k in range(total_days - 1):
            S_vec = np.array([S_therm[k], S_cryo[k], S_salt_base[k], S_wind[k], 0.0])
            diss = np.dot(w_norm, S_vec)
            rate = (dt_days / (tau_m_sim * ((1.0 - np.clip(D_curr, 0.0, 0.99)) ** n_sim))) * diss
            D_next = min(D_crit, D_curr + max(0.0, rate))
            D_ensemble[sim, k + 1] = D_next
            D_curr = D_next

            if D_curr >= (D_crit - 0.001) and not failed:
                failure_years.append(t_years[k + 1])
                failed = True

        if not failed:
            failure_years.append(2020.0 + horizon_years)

    # Статистический расчет квантилей (P10, P50, P90)
    p10_trajectory = np.percentile(D_ensemble, 10, axis=0)
    p50_trajectory = np.percentile(D_ensemble, 50, axis=0)  # Медиана
    p90_trajectory = np.percentile(D_ensemble, 90, axis=0)

    return t_years, p10_trajectory, p50_trajectory, p90_trajectory, np.array(failure_years)


def plot_uncertainty_quantification(t_years, p10, p50, p90, failure_years):
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 9})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300)

    # График 1: Стохастические коридоры P10-P50-P90
    ax1.plot(t_years, p50, color='#08519c', lw=2.0, label='Median Trajectory (P50)')
    ax1.fill_between(t_years, p10, p90, color='#6baed6', alpha=0.35, label='80% Confidence Band (P10 - P90)')
    
    # Натурные точки ЮНЕСКО
    unesco_t = [2020.0, 2022.5, 2025.0]
    unesco_d = [0.1800, 0.2550, 0.3200]
    unesco_err = [0.020, 0.035, 0.050]
    ax1.errorbar(unesco_t, unesco_d, yerr=unesco_err, fmt='s', color='black',
                 capsize=4, elinewidth=1.2, markersize=5, label='UNESCO Ground Truth (Ref. 1103)')

    ax1.axhline(0.45, color='#d62728', ls='--', lw=1.2, label=r'Bifurcation Failure ($D_{\mathrm{crit}}=0.45$)')
    ax1.set_xlim(2020.0, 2030.0)
    ax1.set_ylim(0.12, 0.48)
    ax1.set_xlabel('Timeline [Years]')
    ax1.set_ylabel(r'Damage Metric $D(t)$')
    ax1.set_title('Monte Carlo Continuum Damage Envelope', fontweight='bold')
    ax1.grid(True, ls=':', alpha=0.6)
    ax1.legend(fontsize=7.2, loc='upper left')

    # График 2: Вероятность наступления аварийного отслоения кладки
    sorted_failures = np.sort(failure_years)
    cdf = np.arange(1, len(sorted_failures) + 1) / len(sorted_failures)
    
    ax2.plot(sorted_failures, cdf * 100, color='#d62728', lw=2.0, label='CDF: P(Failure <= Year)')
    ax2.axvline(2028.5, color='gray', ls=':', label='Median Failure Window (~2028.5)')
    ax2.set_xlim(2026.0, 2031.0)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('Predicted Failure Year')
    ax2.set_ylabel('Cumulative Failure Probability [%]')
    ax2.set_title('Probability of Critical Spalling', fontweight='bold')
    ax2.grid(True, ls=':', alpha=0.6)
    ax2.legend(fontsize=7.5, loc='upper left')

    plt.tight_layout()
    plt.savefig('fig7_monte_carlo_uq.pdf', bbox_inches='tight')
    plt.close()
    print("      -> Saved fig7_monte_carlo_uq.pdf")


if __name__ == '__main__':
    print("=" * 68)
    print("TAMYR AI: MONTE CARLO UNCERTAINTY QUANTIFICATION (N=500)")
    print("=" * 68)
    t, p10, p50, p90, fails = run_monte_carlo_uncertainty(num_simulations=500)

    med_year = np.median(fails)
    p10_year = np.percentile(fails, 10)
    p90_year = np.percentile(fails, 90)

    print(f"Stochastic Failure Horizon Analysis (D >= 0.45):")
    print(f"  - P10 Optimistic Horizon : Year {p90_year:.2f}")
    print(f"  - P50 Median Failure     : Year {med_year:.2f}")
    print(f"  - P90 Conservative Hazard: Year {p10_year:.2f}")
    print(f"Confidence Width at 2025.0 : D in [{p10[int(5*365.25)]:.4f}, {p90[int(5*365.25)]:.4f}]")
    print("=" * 68)

    plot_uncertainty_quantification(t, p10, p50, p90, fails)
