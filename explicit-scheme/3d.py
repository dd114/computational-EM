#!/usr/bin/env python3
# fdtd_3d_wg_resonator_pml_viz_with_slices.py
#
# 3D версия с отображением срезов XY, XZ, YZ
#  - Показывает распространение волны в реальном времени
#  - Считает max(E_strip / E_res) за шаги
#  - Строит график зависимости этого соотношения от частоты
#  - Отображает контуры всех областей (PML, WG, Resonator)
#  - 3D геометрия: резонатор = шар, волноводы = цилиндры
#
import os
import math
import time
import numpy as np
import matplotlib

# ----------------------- Настройка matplotlib backend --------------------------
headless = (os.environ.get('DISPLAY') is None and os.environ.get('WAYLAND_DISPLAY') is None)
if headless:
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------- Физические константы ---------------------------------
c0 = 299792458.0
lambda0 = 1.55e-6
f0 = c0 / lambda0
f0 = 1.9341449e14
lambda0 = c0 / f0

# -------------------------- 3D Сетка ----------------------------------------------
Lx = 8.0e-6
Ly = 8.0e-6
Lz = 8.0e-6  # Добавлено третье измерение
dx = 0.05e-6  # Увеличен шаг для снижения памяти
dy = dx
dz = dx
Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1
Nz = int(round(Lz / dz)) + 1
x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)
z = np.linspace(0.0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

n_min = 1.0
v_max = c0 / n_min
CFL = 0.35
dt = CFL * dx / (v_max * math.sqrt(3.0))  # 3D CFL условие
idx2 = 1.0 / (dx * dx)
idy2 = 1.0 / (dy * dy)
idz2 = 1.0 / (dz * dz)

print(f"Сетка: {Nx} x {Ny} x {Nz} = {Nx*Ny*Nz/1e6:.2f} млн ячеек")
print(f"Память для одного поля: {Nx*Ny*Nz*8/1e6:.2f} МБ")

# -------------------------- 3D PML --------------------------------------------------
pml_thickness_m = 0.05 * min(Lx, Ly, Lz)
pml_cells = int(round(pml_thickness_m / dx))
alpha_max = 0.8 / dt
ix = np.arange(Nx).reshape(-1, 1, 1)
jy = np.arange(Ny).reshape(1, -1, 1)
kz = np.arange(Nz).reshape(1, 1, -1)
dist_x = np.minimum(ix, Nx - 1 - ix)
dist_y = np.minimum(jy, Ny - 1 - jy)
dist_z = np.minimum(kz, Nz - 1 - kz)
dist_min = np.minimum(np.minimum(dist_x, dist_y), dist_z).astype(np.float64)
alpha = np.zeros_like(dist_min)
mask_pml = dist_min < pml_cells
alpha[mask_pml] = alpha_max * ((pml_cells - dist_min[mask_pml]) / pml_cells) ** 2

# -------------------------- 3D Геометрия --------------------------------------------
n_bg = 1.0
n_wg = 3.5
n_res = 3.5
n_map = n_bg * np.ones((Nx, Ny, Nz), dtype=np.float64)
contour_mask = np.zeros((Nx, Ny, Nz), dtype=np.float64)

# Верхний волновод (цилиндр вдоль X)
wg_t_radius = 0.2e-6
wg_t_cy = Ly - pml_thickness_m - 2 * wg_t_radius
wg_t_cz = Lz / 2
wg_t_mask = ((Y - wg_t_cy)**2 + (Z - wg_t_cz)**2 <= wg_t_radius**2)
n_map[wg_t_mask] = n_bg
contour_mask[wg_t_mask] = 2.0

# Нижний волновод (цилиндр вдоль X)
wg_b_radius = 0.2e-6
wg_b_cy = pml_thickness_m + 2 * wg_b_radius
wg_b_cz = Lz / 2
wg_b_mask = ((Y - wg_b_cy)**2 + (Z - wg_b_cz)**2 <= wg_b_radius**2)
n_map[wg_b_mask] = n_wg
contour_mask[wg_b_mask] = 3.0

# Резонатор (шар)
res_cx = Lx / 2
res_cy = Ly / 2
res_cz = Lz / 2
overlap_b = 0.1e-6
res_radius = min(res_cx - pml_thickness_m, 
                 res_cy - (wg_b_cy + wg_b_radius) + overlap_b,
                 Lz/2 - pml_thickness_m)
res_mask = (X - res_cx)**2 + (Y - res_cy)**2 + (Z - res_cz)**2 <= res_radius**2
n_map[res_mask] = n_res
contour_mask[res_mask] = 4.0

# Полоса вокруг резонатора (сферическая оболочка)
strip_thickness = 0.05 * res_radius
strip_mask = ((X - res_cx)**2 + (Y - res_cy)**2 + (Z - res_cz)**2 >= (res_radius - strip_thickness)**2) & (res_mask)
n_map[strip_mask] = n_res
contour_mask[strip_mask] = 5.0

# PML
contour_mask[mask_pml] = 1.0

v_map = c0 / n_map
v2_map = v_map ** 2

# -------------------------- 3D Лапласиан -------------------------------------------
def laplacian_3d(A):
    """Вычисляет 3D лапласиан с граничными условиями."""
    L = np.zeros_like(A)
    L[1:-1, 1:-1, 1:-1] = (
        (A[2:,1:-1,1:-1] + A[:-2,1:-1,1:-1] - 2.0*A[1:-1,1:-1,1:-1]) * idx2 +
        (A[1:-1,2:,1:-1] + A[1:-1,:-2,1:-1] - 2.0*A[1:-1,1:-1,1:-1]) * idy2 +
        (A[1:-1,1:-1,2:] + A[1:-1,1:-1,:-2] - 2.0*A[1:-1,1:-1,1:-1]) * idz2
    )
    return L

# -------------------------- Инициализация поля ----------------------------------
def init_field_directed_3d(X, Y, Z, freq, x0=pml_thickness_m, A=0.7, theta_deg=0.0):
    """Создаёт начальный направленный пучок в нижнем волноводе (3D)."""
    lam = c0 / freq
    k0 = 2 * np.pi / lam
    theta = np.deg2rad(theta_deg)
    kx = k0 * np.cos(theta)
    ky = k0 * np.sin(theta)
    Xc = X - x0
    Yc = Y - wg_b_cy
    Zc = Z - wg_b_cz
    envelope = np.exp(-0.5 * ((Yc / (0.5e-6))**2 + (Zc / (0.5e-6))**2))
    phase = kx * Xc + ky * Yc
    E_init = A * envelope * np.cos(phase)
    return np.where(X < 0.9 * Lx / 2, E_init * wg_b_mask, 0)

# -------------------------- Источник --------------------------------------------
src_x_cols = [int(round(1.0e-6 / dx)), int(round(1.0e-6 / dx)) + 1]
src_y0 = wg_b_cy
src_z0 = wg_b_cz
src_sigma = 0.7e-6
src_mask = np.exp(-0.5 * (((Y - src_y0) / src_sigma)**2 + ((Z - src_z0) / src_sigma)**2))
src_amplitude = 0.5
def src_time(t): return 0.0

# -------------------------- Визуализация (3 среза) ----------------------------------------
plt.ion()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Срез XY (фиксированный Z)
slice_z = Nz // 2
im1 = axes[0].imshow(np.zeros((Ny, Nx)).T, origin='lower',
                     extent=[0, Lx*1e6, 0, Ly*1e6],
                     cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                     interpolation='bilinear')
axes[0].set_xlabel('x (µm)')
axes[0].set_ylabel('y (µm)')
axes[0].set_title(f'XY Slice (z={slice_z*dz*1e6:.2f} µm)')

# === Добавляем контуры областей ===
axes[0].contour(X[:, :, 0].T * 1e6, Y[:, :, 0].T * 1e6, (contour_mask[:, :, slice_z] == 1).T,
           levels=[0.5], colors='black', linewidths=1.2, label='PML')
axes[0].contour(X[:, :, 0].T * 1e6, Y[:, :, 0].T * 1e6, (contour_mask[:, :, slice_z] == 2).T,
           levels=[0.5], colors='blue', linewidths=2.5, label='Top WG')
axes[0].contour(X[:, :, 0].T * 1e6, Y[:, :, 0].T * 1e6, (contour_mask[:, :, slice_z] == 3).T,
           levels=[0.5], colors='orange', linewidths=2.5, label='Bottom WG')
axes[0].contour(X[:, :, 0].T * 1e6, Y[:, :, 0].T * 1e6, (contour_mask[:, :, slice_z] == 4).T,
           levels=[0.5], colors='red', linewidths=2.5, label='Resonator')
axes[0].contour(X[:, :, 0].T * 1e6, Y[:, :, 0].T * 1e6, (contour_mask[:, :, slice_z] == 5).T,
           levels=[0.5], colors='green', linewidths=2.5, label='Strip')

# Срез XZ (фиксированный Y)
slice_y = Ny // 2
im2 = axes[1].imshow(np.zeros((Nz, Nx)).T, origin='lower',
                     extent=[0, Lx*1e6, 0, Lz*1e6],
                     cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                     interpolation='bilinear')
axes[1].set_xlabel('x (µm)')
axes[1].set_ylabel('z (µm)')
axes[1].set_title(f'XZ Slice (y={slice_y*dy*1e6:.2f} µm)')

# === Добавляем контуры областей ===
axes[1].contour(X[:, slice_y, :].T * 1e6, Z[:, slice_y, :].T * 1e6, (contour_mask[:, slice_y, :] == 1).T,
           levels=[0.5], colors='black', linewidths=1.2, label='PML')
axes[1].contour(X[:, slice_y, :].T * 1e6, Z[:, slice_y, :].T * 1e6, (contour_mask[:, slice_y, :] == 2).T,
           levels=[0.5], colors='blue', linewidths=2.5, label='Top WG')
axes[1].contour(X[:, slice_y, :].T * 1e6, Z[:, slice_y, :].T * 1e6, (contour_mask[:, slice_y, :] == 3).T,
           levels=[0.5], colors='orange', linewidths=2.5, label='Bottom WG')
axes[1].contour(X[:, slice_y, :].T * 1e6, Z[:, slice_y, :].T * 1e6, (contour_mask[:, slice_y, :] == 4).T,
           levels=[0.5], colors='red', linewidths=2.5, label='Resonator')
axes[1].contour(X[:, slice_y, :].T * 1e6, Z[:, slice_y, :].T * 1e6, (contour_mask[:, slice_y, :] == 5).T,
           levels=[0.5], colors='green', linewidths=2.5, label='Strip')

# Срез YZ (фиксированный X)
slice_x = Nx // 2
im3 = axes[2].imshow(np.zeros((Nz, Ny)).T, origin='lower',
                     extent=[0, Ly*1e6, 0, Lz*1e6],
                     cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                     interpolation='bilinear')
axes[2].set_xlabel('y (µm)')
axes[2].set_ylabel('z (µm)')
axes[2].set_title(f'YZ Slice (x={slice_x*dx*1e6:.2f} µm)')

# === Добавляем контуры областей ===
axes[2].contour(Y[slice_x, :, :].T * 1e6, Z[slice_x, :, :].T * 1e6, (contour_mask[slice_x, :, :] == 1).T,
           levels=[0.5], colors='black', linewidths=1.2, label='PML')
axes[2].contour(Y[slice_x, :, :].T * 1e6, Z[slice_x, :, :].T * 1e6, (contour_mask[slice_x, :, :] == 2).T,
           levels=[0.5], colors='blue', linewidths=2.5, label='Top WG')
axes[2].contour(Y[slice_x, :, :].T * 1e6, Z[slice_x, :, :].T * 1e6, (contour_mask[slice_x, :, :] == 3).T,
           levels=[0.5], colors='orange', linewidths=2.5, label='Bottom WG')
axes[2].contour(Y[slice_x, :, :].T * 1e6, Z[slice_x, :, :].T * 1e6, (contour_mask[slice_x, :, :] == 4).T,
           levels=[0.5], colors='red', linewidths=2.5, label='Resonator')
axes[2].contour(Y[slice_x, :, :].T * 1e6, Z[slice_x, :, :].T * 1e6, (contour_mask[slice_x, :, :] == 5).T,
           levels=[0.5], colors='green', linewidths=2.5, label='Strip')

plt.tight_layout()
cbar = fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.02, pad=0.05)
cbar.set_label('E amplitude')

time_text = fig.text(0.5, 0.02, s='', ha='center', color='black', fontsize=12)

# -------------------------- Параметры ------------------------------------------
freqs = [i * f0 / 10 for i in range(10, 105, 5)]
nsteps_measure = 5000  # Уменьшено для 3D (медленнее)
vis_interval = 10
print_interval = 100

# -------------------------- Основная функция симуляции --------------------------
def run_simulation_3d(freq, i, v2_map):
    """Запуск 3D FDTD для одной частоты, с визуализацией и измерением соотношения энергий."""
    E = init_field_directed_3d(X, Y, Z, freq)
    W = np.zeros_like(E)
    top_mask = wg_t_mask.astype(float)
    res_mask_f = res_mask.astype(float)
    strip_mask_f = strip_mask.astype(float)

    output = np.zeros(nsteps_measure)
    max_ratio = 0.0
    
    for n in range(nsteps_measure):
        t = n * dt
        lapE = laplacian_3d(E)
        W_new = W + dt * (v2_map * lapE - alpha * W)
        
        # inj = src_time(t) * src_amplitude
        # for col in src_x_cols:
        #     if 0 <= col < Nx:
        #         W_new[col, :, :] += inj * src_mask[col, :, :]

        E_new = E + dt * W_new
        # Граничные условия (нулевые на всех границах + PML)
        E_new[0,:,:]=0; E_new[-1,:,:]=0
        E_new[:,0,:]=0; E_new[:,-1,:]=0
        E_new[:,:,0]=0; E_new[:,:,-1]=0
        W_new[0,:,:]=0; W_new[-1,:,:]=0
        W_new[:,0,:]=0; W_new[:,-1,:]=0
        W_new[:,:,0]=0; W_new[:,:,-1]=0
        E, W = E_new, W_new

        # Энергии
        ratio = np.sum((E**2) * strip_mask_f) / (np.sum((E**2) * res_mask_f) + 1e-15)
        output[n] = ratio

        # Вывод в консоль
        if n % print_interval == 0:
            mem_gb = E.nbytes / 1e9
            print(f"[i={i}/{len(freqs)} f={freq/1e12:.1f} THz] step {n}/{nsteps_measure} ratio={ratio:.6f} mem={mem_gb:.2f}GB")

        # Закрытие входа нижнего волновода
        if n == 2000:
            n_map[wg_b_mask] = n_bg
            v_map = c0 / n_map
            v2_map = v_map ** 2

        # Обновление анимации (3 среза)
        if n % vis_interval == 0:
            im1.set_data(E[:, :, slice_z].T)  # Срез XY
            im2.set_data(E[:, slice_y, :].T)  # Срез XZ
            im3.set_data(E[slice_x, :, :].T)  # Срез YZ
            
            amp = max(1e-12, np.max(np.abs(E)))
            im1.set_clim(-0.9*amp, 0.9*amp)
            im2.set_clim(-0.9*amp, 0.9*amp)
            im3.set_clim(-0.9*amp, 0.9*amp)
            
            time_text.set_text(f"f={freq/1e12:.1f} THz, t={t*1e15:.1f} fs, step={n}")
            plt.pause(0.001)

    return output

# -------------------------- Основной цикл по частотам --------------------------
max_ratios = []
overall_start = time.time()
print("\n" + "="*60)
print("ЗАПУСК 3D FDTD СИМУЛЯЦИИ")
print("="*60)

# Для демонстрации запускаем только несколько частот
test_freqs = freqs[-3:]  # Последние 3 частоты для теста
for i, f in enumerate(test_freqs):
    print(f"\n === Frequency = {f/1e12:.2f} THz ===")
    output = run_simulation_3d(f, i, v2_map)
    max_ratios.append(output[2500:].mean())  
    
overall_end = time.time()

print(f"\n✅ Completed {len(test_freqs)} frequencies in {overall_end - overall_start:.2f} s")

# -------------------------- График зависимости ---------------------------------
plt.ioff()
plt.figure(figsize=(8,5))
plt.plot(np.array(test_freqs)/1e12, max_ratios, 'o-', linewidth=2, markersize=8)
plt.xlabel('Frequency (THz)')
plt.ylabel('Max ratio E_strip / E_resonator')
plt.title('Dependence of max energy ratio on frequency (3D)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=True)