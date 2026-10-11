"""
Tamyr AI - Native Desktop 3D Masonry Inspector
Interactive 3D damage visualization using Matplotlib (no external web services).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Сетка геометрии фрагмента цоколя Ясави
Nx, Ny, Nz = 18, 12, 16
Lx, Ly, Lz = 0.25, 0.50, 1.00  # метры (глубина, ширина, высота)

xs = np.linspace(0.0, Lx, Nx)
ys = np.linspace(0.0, Ly, Ny)
zs = np.linspace(0.0, Lz, Nz)

X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

# 2. Физическое поле деградации D(x, y, z)
# Пограничный слой: затухание вглубь стены (по оси X)
f_depth = np.exp(-X / 0.08)
# Капиллярный подсос грунтовых солей снизу (по оси Z)
f_capillary = np.exp(-Z / 0.40)

D_base = 0.1800
Damage_D = np.clip(D_base + 0.1658 * f_depth * (0.65 + 0.35 * f_capillary), 0.15, 0.45)
Elasticity_E = 4.0 * (1.0 - Damage_D)  # ГПа

# 3. Фильтрация точек для визуализации (показываем фасад, основание и срезы)
# Чтобы объем не превращался в сплошную кашу, визуализируем градиентные слои
mask = (X <= 0.08) | (Z <= 0.25) | (X >= 0.20) | (Y == ys[0]) | (Y == ys[-1])

x_plot = X[mask].flatten()
y_plot = Y[mask].flatten()
z_plot = Z[mask].flatten()
d_plot = Damage_D[mask].flatten()
e_plot = Elasticity_E[mask].flatten()

# 4. Отрисовка интерактивной 3D сцены
fig = plt.figure(figsize=(10, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    x_plot * 100,  # переводим глубину в сантиметры
    y_plot * 100,  # ширину в см
    z_plot * 100,  # высоту в см
    c=d_plot,
    cmap='turbo',
    s=28,
    alpha=0.85,
    edgecolors='none'
)

cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
cbar.set_label('Damage Metric D(t) [0 - Intact, 0.45 - Spalling]', rotation=90, labelpad=15)

ax.set_title('Tamyr AI: 3D Heterogeneous Masonry Damage Field (2025)\n(Drag mouse to rotate, right-click to zoom)',
             fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel('Wall Depth X [cm] (0 = Facade)')
ax.set_ylabel('Width Y [cm]')
ax.set_zlabel('Height Z [cm] (0 = Foundation)')

# Начальный угол обзора
ax.view_init(elev=25, azim=-55)

print("\n[OK] 3D-модель запущена!")
print("     - Зажмите ЛЕВУЮ кнопку мыши и двигайте курсор, чтобы крутить куб кладки.")
print("     - Зажмите ПРАВУЮ кнопку мыши для приближения/отдаления.")
plt.tight_layout()
plt.show()
