# fdtd_wg_resonator_pml_viz_3d_cn.py
#
# 3D версия с консервативной схемой Кранка-Николсона для волнового уравнения.
# E = (0, 0, E_z(x, y, z)) - скалярное 3D волновое уравнение
#  - Использует неявную схему (требуется решение СЛАУ на каждом шаге)
#  - 3D разреженная матрица Лапласиана
#  - Визуализация через 2D срезы (xy, xz, yz плоскости)
#  - Уменьшенная сетка для демонстрации (память и скорость)
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

# ----------------------- Сторонние библиотеки для линейной алгебры ---------------
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# -------------------------- Физические константы ---------------------------------
c0 = 299792458.0
lambda0 = 1.55e-6
f0 = c0 / lambda0
f0 = 1.9341449e14
lambda0 = c0 / f0

# -------------------------- 3D Сетка -------------------------------------------
Lx = 8.0e-6
Ly = 6.0e-6
Lz = 6.0e-6
dx = 0.05e-6 * 5
dy = dx
dz = dx
Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1
Nz = int(round(Lz / dz)) + 1
x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)
z = np.linspace(0.0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

print(f"3D Grid: {Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} points")

n_min = 1.0
v_max = c0 / n_min
CFL = 0.35
dt = CFL * dx / (v_max * math.sqrt(3.0))  # sqrt(3) for 3D
idx2 = 1.0 / (dx * dx)

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
n_wg = 4.0
n_res = 4.5
n_map = n_bg * np.ones((Nx, Ny, Nz), dtype=np.float64)
contour_mask = np.zeros((Nx, Ny, Nz), dtype=np.float64)

# Волновод вдоль оси X (в плоскости z = Lz/2)
wg_half_h = 0.15e-6
wg_cy = Ly / 2
wg_cz = Lz / 2
wg_mask = (np.abs(Y - wg_cy) <= wg_half_h) & (np.abs(Z - wg_cz) <= wg_half_h)
n_map[wg_mask] = n_wg
contour_mask[wg_mask] = 2.0

# Резонатор (сфера)
res_cx = Lx * 0.6
res_cy = Ly / 2
res_cz = Lz / 2
res_radius = 0.8e-6
res_mask = (X - res_cx)**2 + (Y - res_cy)**2 + (Z - res_cz)**2 <= res_radius**2
n_map[res_mask] = n_res
contour_mask[res_mask] = 3.0

# Полоса вокруг резонатора
strip_thickness = 0.15 * res_radius
strip_mask = ((X - res_cx)**2 + (Y - res_cy)**2 + (Z - res_cz)**2 >= (res_radius - strip_thickness)**2) & (res_mask)
n_map[strip_mask] = n_res
contour_mask[strip_mask] = 4.0

# PML контур
contour_mask[mask_pml] = 1.0

v_map = c0 / n_map
v2_map = v_map ** 2

# -------------------------- 3D Лапласиан (разреженная матрица) --------------------
def build_3d_laplacian_matrix(Nx, Ny, Nz, dx):
    """Создает разреженную матрицу 3D оператора Лапласа с Дирихле ГУ."""
    N = Nx * Ny * Nz
    main_diag = -6.0 / (dx * dx) * np.ones(N)
    off_diag = 1.0 / (dx * dx) * np.ones(N)
    
    # Индексы: k = i * Ny * Nz + j * Nz + l (C-order, indexing='ij' в meshgrid)
    # Соседи: +/-1 (z), +/-Nz (y), +/-Ny*Nz (x)
    
    strides_z = 1
    strides_y = Nz
    strides_x = Ny * Nz
    
    # Обнуляем связи на границах
    for i in range(Nx):
        for j in range(Ny):
            for l in range(Nz):
                idx = i * strides_x + j * strides_y + l * strides_z
                
                # Границы по Z
                if l == 0 or l == Nz - 1:
                    off_diag[idx] = 0
                    if l > 0:
                        off_diag[idx - 1] = 0
                
                # Границы по Y
                if j == 0 or j == Ny - 1:
                    off_diag[idx] = 0
                    if j > 0:
                        off_diag[idx - strides_y] = 0
                
                # Границы по X
                if i == 0 or i == Nx - 1:
                    off_diag[idx] = 0
                    if i > 0:
                        off_diag[idx - strides_x] = 0
    
    # Для эффективности используем другой подход - диагонали
    offsets = [0, strides_z, -strides_z, strides_y, -strides_y, strides_x, -strides_x]
    
    # Создаем диагонали
    diags_list = [main_diag]
    for offset in offsets[1:]:
        d = np.ones(N) * (1.0 / (dx * dx))
        # Обнуляем связи на границах для каждого направления
        if offset == strides_z:  # +z
            d[Nx*Ny*(Nz-1):] = 0  # последняя плоскость по z
        elif offset == -strides_z:  # -z
            d[:Nx*Ny] = 0  # первая плоскость по z
        elif offset == strides_y:  # +y
            for i in range(Nx):
                d[i*strides_x + (Ny-1)*strides_y : i*strides_x + Ny*strides_y] = 0
        elif offset == -strides_y:  # -y
            for i in range(Nx):
                d[i*strides_x : i*strides_x + strides_y] = 0
        elif offset == strides_x:  # +x
            d[(Nx-1)*strides_x:] = 0
        elif offset == -strides_x:  # -x
            d[:strides_x] = 0
        diags_list.append(d)
    
    L_mat = sp.diags(diags_list, offsets, shape=(N, N), format='csc')
    return L_mat

print("Building 3D Laplacian matrix...")
L_mat = build_3d_laplacian_matrix(Nx, Ny, Nz, dx)
print(f"Laplacian matrix: {L_mat.nnz:,} non-zero elements")

# -------------------------- Инициализация 3D поля ----------------------------------
def init_field_3d(X, Y, Z, freq, x0=1.0e-6, A=0.7):
    """Создаёт начальный направленный пучок в волноводе."""
    lam = c0 / freq
    k0 = 2 * np.pi / lam
    kx = k0
    Xc = X - x0
    Yc = Y - wg_cy
    Zc = Z - wg_cz
    envelope = np.exp(-0.5 * (Yc / (0.3e-6))**2) * np.exp(-0.5 * (Zc / (0.3e-6))**2)
    phase = kx * Xc
    E_init = A * envelope * np.cos(phase)
    return np.where(X < Lx * 0.4, E_init * wg_mask, 0)

# -------------------------- 3D Источник --------------------------------------------
src_x_col = int(round(1.0e-6 / dx))
src_sigma = 0.3e-6
src_amplitude = 0.5
def src_time(t): return 0.0

# -------------------------- Визуализация (2D срезы) ----------------------------------------
plt.ion()
fig = plt.figure(figsize=(14, 10))

# Три среза: XY (при z=Lz/2), XZ (при y=Ly/2), YZ (при x=Lx/2)
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

slice_z = Nz // 2
slice_y = Ny // 2
slice_x = Nx // 2

im1 = ax1.imshow(np.zeros((Ny, Nx)).T, origin='lower',
                 extent=[0, Lx*1e6, 0, Ly*1e6],
                 cmap='RdBu_r', vmin=-0.5, vmax=0.5, interpolation='bilinear')
ax1.set_xlabel('x (µm)')
ax1.set_ylabel('y (µm)')
ax1.set_title(f'E_z at z = {z[slice_z]*1e6:.2f} µm')

im2 = ax2.imshow(np.zeros((Nz, Nx)).T, origin='lower',
                 extent=[0, Lx*1e6, 0, Lz*1e6],
                 cmap='RdBu_r', vmin=-0.5, vmax=0.5, interpolation='bilinear')
ax2.set_xlabel('x (µm)')
ax2.set_ylabel('z (µm)')
ax2.set_title(f'E_z at y = {y[slice_y]*1e6:.2f} µm')

im3 = ax3.imshow(np.zeros((Nz, Ny)).T, origin='lower',
                 extent=[0, Ly*1e6, 0, Lz*1e6],
                 cmap='RdBu_r', vmin=-0.5, vmax=0.5, interpolation='bilinear')
ax3.set_xlabel('y (µm)')
ax3.set_ylabel('z (µm)')
ax3.set_title(f'E_z at x = {x[slice_x]*1e6:.2f} µm')

# График энергии
ax4.set_xlabel('Step')
ax4.set_ylabel('Energy Ratio')
ax4.set_title('E_strip / E_resonator')
ax4.grid(True)
line4, = ax4.plot([], [], 'b-', linewidth=2)
ax4_data_x = []
ax4_data_y = []

time_text = fig.text(0.01, 0.99, s='', transform=fig.transFigure, color='black', fontsize=10, verticalalignment='top')

fig.tight_layout()

# -------------------------- Параметры ------------------------------------------
freqs = [f0 * 0.8, f0 * 0.9, f0 * 1.0, f0 * 1.1, f0 * 1.2]
nsteps_measure = 1500  # Уменьшено для 3D (очень медленно)
vis_interval = 20
print_interval = 100
geometry_change_step = 800

# -------------------------- Основная функция 3D симуляции --------------------------
def run_simulation_3d(freq, i, v2_map_init):
    """Запуск 3D симуляции с схемой Кранка-Николсона."""
    
    global ax4_data_x, ax4_data_y
    ax4_data_x = []
    ax4_data_y = []
    
    # 1. Инициализация полей
    E = init_field_3d(X, Y, Z, freq)
    W = np.zeros_like(E)
    
    # Маски
    res_mask_f = res_mask.astype(float)
    strip_mask_f = strip_mask.astype(float)
    
    N = Nx * Ny * Nz
    I_mat = sp.eye(N, format='csc')
    
    # Диагональные матрицы
    diag_alpha = sp.diags(alpha.ravel(), 0, format='csc')
    diag_v2 = sp.diags(v2_map_init.ravel(), 0, format='csc')
    
    # Оператор LHS: M = I + 0.5*dt*alpha - 0.25*dt^2*v^2*Laplacian
    print(f"  Building system matrix (N={N:,})...")
    M = I_mat + 0.5 * dt * diag_alpha - 0.25 * dt**2 * (diag_v2 @ L_mat)
    
    print(f"  Factorizing matrix (this may take a while)...")
    t0 = time.time()
    try:
        solver = spla.factorized(M.tocsc())
    except Exception as e:
        print(f"  Error factorizing: {e}")
        return np.zeros(nsteps_measure)
    t1 = time.time()
    print(f"  Factorization took {t1-t0:.2f} s")
    
    output = np.zeros(nsteps_measure)
    current_v2_map = v2_map_init.copy()
    matrix_updated = False
    
    # Граничные индексы
    boundary_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[-1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    boundary_indices = np.where(boundary_mask.ravel())[0]
    
    for n in range(nsteps_measure):
        t = n * dt
        
        # Изменение геометрии
        if n == geometry_change_step and not matrix_updated:
            n_map[wg_mask] = n_bg
            v_map = c0 / n_map
            current_v2_map = v_map ** 2
            
            diag_v2_new = sp.diags(current_v2_map.ravel(), 0, format='csc')
            M_new = I_mat + 0.5 * dt * diag_alpha - 0.25 * dt**2 * (diag_v2_new @ L_mat)
            solver = spla.factorized(M_new.tocsc())
            matrix_updated = True
            print(f"  [Step {n}] Geometry updated, matrix refactored.")
        
        # Векторизация
        E_flat = E.ravel()
        W_flat = W.ravel()
        
        # Лапласиан
        LapE = L_mat @ E_flat
        LapW = L_mat @ W_flat
        
        v2_flat = current_v2_map.ravel()
        alpha_flat = alpha.ravel()
        
        # RHS
        term_W = W_flat - 0.5 * dt * alpha_flat * W_flat + 0.25 * dt**2 * v2_flat * LapW
        term_E = dt * v2_flat * LapE
        rhs = term_W + term_E
        
        # Источник
        inj = src_time(t) * src_amplitude
        if inj != 0.0:
            idx_start = src_x_col * Ny * Nz
            idx_end = (src_x_col + 1) * Ny * Nz
            src_profile = inj * np.exp(-0.5 * ((Y[:, :, 0] - wg_cy) / src_sigma)**2) * \
                               np.exp(-0.5 * ((Z[:, :, 0] - wg_cz) / src_sigma)**2)
            rhs[idx_start:idx_end] += src_profile.ravel()
        
        # Решение
        try:
            W_new_flat = solver(rhs)
        except Exception:
            W_new_flat = rhs
        
        # Обновление E
        E_new_flat = E_flat + 0.5 * dt * (W_new_flat + W_flat)
        
        # Граничные условия
        E_new_flat[boundary_indices] = 0.0
        W_new_flat[boundary_indices] = 0.0
        
        E = E_new_flat.reshape((Nx, Ny, Nz))
        W = W_new_flat.reshape((Nx, Ny, Nz))
        
        # Измерения
        energy_strip = np.sum((E**2) * strip_mask_f)
        energy_res = np.sum((E**2) * res_mask_f)
        ratio = energy_strip / (energy_res + 1e-12)
        output[n] = ratio
        
        if n % print_interval == 0:
            print(f"[i={i}/{len(freqs)} f={freq/1e12:.1f} THz] step {n}/{nsteps_measure} ratio={ratio:.6f}")
        
        # Визуализация
        if n % vis_interval == 0:
            im1.set_data(E[:, :, slice_z].T)
            im2.set_data(E[:, slice_y, :].T)
            im3.set_data(E[slice_x, :, :].T)
            
            amp = max(1e-12, np.max(np.abs(E)))
            im1.set_clim(-0.9*amp, 0.9*amp)
            im2.set_clim(-0.9*amp, 0.9*amp)
            im3.set_clim(-0.9*amp, 0.9*amp)
            
            ax4_data_x.append(n)
            ax4_data_y.append(ratio)
            line4.set_data(ax4_data_x, ax4_data_y)
            ax4.relim()
            ax4.autoscale_view()
            
            time_text.set_text(f"f={freq/1e12:.1f} THz, t={t*1e15:.1f} fs, step={n}")
            plt.pause(0.001)
    
    return output

# -------------------------- Основной цикл по частотам --------------------------
max_ratios = []
overall_start = time.time()

print(f"\n{'='*60}")
print(f"3D Crank-Nicolson Wave Simulation")
print(f"{'='*60}")
print(f"Grid: {Nx} x {Ny} x {Nz}")
print(f"Total points: {Nx*Ny*Nz:,}")
print(f"Time step: dt = {dt*1e15:.3f} fs")
print(f"{'='*60}\n")

for i, f in enumerate(freqs):
    print(f"\n{'='*60}")
    print(f"Frequency {i+1}/{len(freqs)}: {f/1e12:.2f} THz")
    print(f"{'='*60}")
    output = run_simulation_3d(f, i, v2_map)
    start_idx = int(nsteps_measure * 0.5)
    max_ratios.append(output[start_idx:].mean())

overall_end = time.time()

print(f"\n{'='*60}")
print(f"✅ Completed {len(freqs)} frequencies in {overall_end - overall_start:.2f} s")
print(f"{'='*60}\n")

# -------------------------- График зависимости ---------------------------------
plt.ioff()
plt.figure(figsize=(8, 5))
plt.plot(np.array(freqs)/1e12, max_ratios, 'o-', linewidth=2, color='darkgreen', markersize=8)
plt.xlabel('Frequency (THz)')
plt.ylabel('Avg Energy Ratio (Strip/Resonator)')
plt.title('3D: Energy Ratio vs Frequency (Crank-Nicolson)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show(block=True)