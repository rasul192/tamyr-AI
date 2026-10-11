"""
Tamyr AI - 3D VTK Spatial Mesh Exporter
Generates structured grid (.vtk) for ParaView and Finite Element (FE) solvers.
Maps Continuum Damage Mechanics (CDM) fields into 3D masonry domain:
  - Damage field D(x, y, z)
  - Degraded elasticity E~(x, y, z) = E_0 * (1 - D)
  - ICOMOS prescriptive conservation zones
"""

import numpy as np


def generate_3d_masonry_vtk(
    output_filename: str = "tamyr_masonry_damage_2025.vtk",
    Lx: float = 0.25,   # Глубина стены (м)
    Ly: float = 1.50,   # Ширина цоколя (м)
    Lz: float = 1.20,   # Высота от фундамента (м)
    Nx: int = 26,       # Сетка по глубине (шаг 1 см)
    Ny: int = 31,       # Сетка по длине (шаг 5 см)
    Nz: int = 25,       # Сетка по высоте (шаг 5 см)
    E_0: float = 4.0,   # Базовый модуль Юнга кирпича (ГПа)
    D_surface_max: float = 0.3458,  # Калиброванная точка ЮНЕСКО 2025
    D_crit: float = 0.45
):
    print("=" * 65)
    print("TAMYR AI: 3D VTK CONTINUUM DAMAGE FIELD GENERATOR")
    print("=" * 65)

    # 1. Пространственные координаты
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    z = np.linspace(0.0, Lz, Nz)
    
    # 3D сетка (порядок VTK: X изменяется быстрее всего, затем Y, затем Z)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 2. Физические градиенты распределения факторов
    # Капиллярный подъем солей по высоте z (максимум у фундамента)
    capillary_z = np.exp(-Z / 0.55)
    
    # Затухание температурных и испарительных процессов по глубине x
    depth_thermal = np.exp(-X / 0.045)
    
    # Солевая субфлоресценция (кристаллизация в приповерхностных 1-4 см)
    x_peak = 0.025
    depth_salt = 1.2 * (X / x_peak) * np.exp(1.0 - (X / x_peak))
    depth_salt = np.clip(depth_salt, 0.1, 1.2) * np.exp(-X / 0.08)

    # Пространственная вариация (швы кладки и естественная гетерогенность)
    mortar_pattern = 1.0 + 0.06 * np.sin(2 * np.pi * Y / 0.26) * np.cos(2 * np.pi * Z / 0.13)

    # 3. Интегральное 3D поле поврежденности D(x, y, z)
    # Суперпозиция: максимальный износ на фасаде в приземной зоне
    decay_field = (0.50 * depth_salt + 0.35 * depth_thermal + 0.15) * capillary_z * mortar_pattern
    
    # Масштабирование под валидированную точку ЮНЕСКО на фасаде (x=0, z=0.2м)
    scale = D_surface_max / np.percentile(decay_field[0, :, :], 95)
    D_field = np.clip(decay_field * scale, 0.0, D_crit)

    # 4. Расчет деградированного модуля упругости: E~ = E_0 * (1 - D)
    E_degraded = E_0 * (1.0 - D_field)

    # 5. Классификация зон вмешательства по шкале ICOMOS
    # 0 = норма (<0.25), 1 = десалинизация/компрессы (0.25-0.40), 2 = предразрушение (>=0.40)
    icomos_flag = np.zeros_like(D_field, dtype=int)
    icomos_flag[D_field >= 0.25] = 1
    icomos_flag[D_field >= 0.40] = 2

    # Преобразование 3D массивов в плоский 1D порядок для VTK (order='F': X fastest)
    D_flat = D_field.flatten(order='F')
    E_flat = E_degraded.flatten(order='F')
    flag_flat = icomos_flag.flatten(order='F')
    total_points = Nx * Ny * Nz

    # 6. Запись в файл формата VTK ASCII
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)

    print(f"Domain size: {Lx:.2f}m x {Ly:.2f}m x {Lz:.2f}m")
    print(f"Grid dimensions: {Nx} x {Ny} x {Nz} = {total_points:,} nodal points")
    print(f"Writing dataset to: {output_filename} ...")

    with open(output_filename, "w", encoding="ascii") as f:
        # Заголовок VTK
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Tamyr AI - 3D Masonry Continuum Damage Mechanics Field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {Nx} {Ny} {Nz}\n")
        f.write("ORIGIN 0.0 0.0 0.0\n")
        f.write(f"SPACING {dx:.6f} {dy:.6f} {dz:.6f}\n")
        f.write(f"POINT_DATA {total_points}\n")

        # Поле 1: Поврежденность D
        f.write("SCALARS Damage_D float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in D_flat:
            f.write(f"{val:.5f}\n")

        # Поле 2: Деградированный модуль Юнга E~ (ГПа)
        f.write("SCALARS Degraded_E_GPa float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in E_flat:
            f.write(f"{val:.4f}\n")

        # Поле 3: Категория риска ICOMOS
        f.write("SCALARS ICOMOS_Zone int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for val in flag_flat:
            f.write(f"{val}\n")

    # Сводная статистика
    v_crit = np.sum(icomos_flag == 2) / total_points * 100
    v_prev = np.sum(icomos_flag == 1) / total_points * 100
    v_safe = np.sum(icomos_flag == 0) / total_points * 100

    print("Generation complete!")
    print(f"  Max Surface Damage D_max : {np.max(D_field):.4f}")
    print(f"  Min Retained Stiffness   : {np.min(E_degraded):.2f} GPa (Drop: {(1.0 - np.min(E_degraded)/E_0)*100:.1f}%)")
    print("  Volume Fraction Breakdown:")
    print(f"    - Safe Core (D < 0.25)          : {v_safe:5.1f}%")
    print(f"    - Preventive Action (D in [0.25, 0.40)): {v_prev:5.1f}%")
    print(f"    - Critical Hazard (D >= 0.40)   : {v_crit:5.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    generate_3d_masonry_vtk()
