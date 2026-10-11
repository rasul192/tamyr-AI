"""
Tamyr AI - 3D Structural Mesh & Damage Field Exporter
Generates standard Legacy VTK (ASCII) structured volume meshes compatible
with Kitware Glance, ParaView, and Finite Element (FEA) solvers.
"""

import numpy as np


def compute_coupled_damage_field(x, y, z):
    """
    Вычисляет пространственное поле повреждений D(x, y, z) кладки цоколя:
    - x: глубина в толщу стены (0.0 м - фасад, 0.25 м - ядро)
    - y: ширина фрагмента фасада
    - z: высота от отмостки (0.0 м - цоколь с солями, 1.0 м - верхняя зона)
    """
    # 1. Затухание климатических пиков вглубь кладки (пограничный слой ~8 см)
    f_depth = np.exp(-x / 0.08)

    # 2. Капиллярный подсос грунтовой влаги и сульфатов снизу (высота капиллярного подъема ~0.6 м)
    f_capillary = np.exp(-z / 0.40)

    # Фоновое повреждение кирпича на 2020 год (D_0 = 0.18)
    D_base = 0.1800

    # Амплитуда деградации фасада к 2025 году
    D_facade_increment = 0.1658 * f_depth * (0.65 + 0.35 * f_capillary)

    D_val = np.clip(D_base + D_facade_increment, 0.1500, 0.4500)
    E_val = 4.0 * (1.0 - D_val)  # Остаточный модуль упругости, ГПа

    return D_val, E_val


def generate_3d_masonry_vtk(
    filename: str = "tamyr_masonry_damage_2025.vtk",
    Nx: int = 26,  # 26 узлов по глубине (шаг 1 см: 0.00..0.25 м)
    Ny: int = 21,  # 21 узел по ширине (шаг 2.5 см: 0.00..0.50 м)
    Nz: int = 21,  # 21 узел по высоте (шаг 5 см: 0.00..1.00 м)
    Lx: float = 0.25,
    Ly: float = 0.50,
    Lz: float = 1.00,
):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)
    total_points = Nx * Ny * Nz

    # Генерация пространственных координат
    xs = np.linspace(0.0, Lx, Nx)
    ys = np.linspace(0.0, Ly, Ny)
    zs = np.linspace(0.0, Lz, Nz)

    # Формирование массивов строго по порядку обхода VTK (X - внутренний, Z - внешний)
    damage_data = []
    elasticity_data = []

    for k in range(Nz):
        z_curr = zs[k]
        for j in range(Ny):
            y_curr = ys[j]
            for i in range(Nx):
                x_curr = xs[i]
                d, e = compute_coupled_damage_field(x_curr, y_curr, z_curr)
                damage_data.append(f"{d:.4f}")
                elasticity_data.append(f"{e:.4f}")

    # Запись строго в формате Legacy VTK 3.0 ASCII с переносами строк LF (\n)
    with open(filename, "w", encoding="utf-8", newline="\n") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Tamyr AI Yasavi Mausoleum Masonry Damage Field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {Nx} {Ny} {Nz}\n")
        f.write("ORIGIN 0.0 0.0 0.0\n")
        f.write(f"SPACING {dx:.5f} {dy:.5f} {dz:.5f}\n")
        f.write(f"POINT_DATA {total_points}\n")

        # Скалярное поле 1: Поврежденность материала D (Kachanov-Lemaitre)
        f.write("SCALARS Damage_D float 1\n")
        f.write("LOOKUP_TABLE default\n")
        # Разбивка по 8 значений в строке для надежности веб-парсеров
        for idx in range(0, total_points, 8):
            f.write(" ".join(damage_data[idx : idx + 8]) + "\n")

        # Скалярное поле 2: Эффективный модуль упругости Degraded_E_GPa
        f.write("SCALARS Degraded_E_GPa float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for idx in range(0, total_points, 8):
            f.write(" ".join(elasticity_data[idx : idx + 8]) + "\n")

    print(f"[OK] 3D Mesh successfully written: {filename}")
    print(f"     Grid Dimensions: {Nx}x{Ny}x{Nz} ({total_points} total nodes)")
    print(f"     Domain Depth X : [0.0, {Lx:.2f}] m (dx = {dx*100:.1f} cm)")
    print(f"     Domain Width Y : [0.0, {Ly:.2f}] m")
    print(f"     Domain Height Z: [0.0, {Lz:.2f}] m")


if __name__ == "__main__":
    generate_3d_masonry_vtk()
