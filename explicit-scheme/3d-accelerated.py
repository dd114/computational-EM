import os
import math
import time
import numpy as np
import matplotlib

# ------------------------ Бэкенд matplotlib ---------------------------------
headless = (os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None)
if headless:
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

# ------------------------ Опционально: Numba --------------------------------
try:
    from numba import njit, prange, set_num_threads, get_num_threads
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    njit = None
    prange = range

    def set_num_threads(n):
        return None

    def get_num_threads():
        return 1

if HAVE_NUMBA:
    set_num_threads(max(1, os.cpu_count() or 1))

# ------------------------ Параметры ускорения --------------------------------
USE_VISUALIZATION = (not headless)  # выключите для максимальной скорости
RATIO_INTERVAL = 1
VIS_INTERVAL = 25
PRINT_INTERVAL = 100

# ------------------------ Константы ------------------------------------------
# c0 = np.float64(299792458.0)
c0 = float(300e+6)  # для удобства, чтобы не было слишком маленьких чисел

# lambda0 = np.float32(1.55e-6)
# f0 = np.float32(c0 / lambda0)

# f0 = np.float64(1.9341449e14)
f0 = float(1e+14) 
lambda0 = float(c0 / f0)

# ------------------------ Сетка ----------------------------------------------
Lx = np.float32(12.0e-6)
Ly = np.float32(12.0e-6)
Lz = np.float32(12.0e-6)

dx = np.float32(0.05e-6)
dy = dx
dz = dx

Nx = int(round(float(Lx / dx))) + 1
Ny = int(round(float(Ly / dy))) + 1
Nz = int(round(float(Lz / dz))) + 1

x = (np.arange(Nx, dtype=np.float32) * dx).astype(np.float32)
y = (np.arange(Ny, dtype=np.float32) * dy).astype(np.float32)
z = (np.arange(Nz, dtype=np.float32) * dz).astype(np.float32)

n_min = np.float32(1.0)
v_max = c0 / n_min
CFL = np.float32(0.35)
dt = np.float32(CFL * dx / (v_max * math.sqrt(3.0)))

idx2 = np.float32(1.0 / (dx * dx))
idy2 = np.float32(1.0 / (dy * dy))
idz2 = np.float32(1.0 / (dz * dz))

print(f"Сетка: {Nx} x {Ny} x {Nz} = {Nx * Ny * Nz / 1e6:.2f} млн ячеек")
print(f"Память для одного поля float32: {Nx * Ny * Nz * 4 / 1e6:.2f} МБ")
if HAVE_NUMBA:
    print(f"Numba включена, потоков: {get_num_threads()}")
else:
    print("Numba не найдена, запасной путь будет заметно медленнее")

# ------------------------ PML -------------------------------------------------
pml_thickness_m = np.float32(0.05 * min(float(Lx), float(Ly), float(Lz)))
pml_cells = int(round(float(pml_thickness_m / dx)))
alpha_max = np.float32(0.8 / float(dt))

ix = np.arange(Nx, dtype=np.int32)[:, None, None]
jy = np.arange(Ny, dtype=np.int32)[None, :, None]
kz = np.arange(Nz, dtype=np.int32)[None, None, :]

dist_x = np.minimum(ix, Nx - 1 - ix)
dist_y = np.minimum(jy, Ny - 1 - jy)
dist_z = np.minimum(kz, Nz - 1 - kz)
dist_min = np.minimum(np.minimum(dist_x, dist_y), dist_z).astype(np.float32)

alpha = np.zeros((Nx, Ny, Nz), dtype=np.float32)
mask_pml = dist_min < pml_cells
alpha[mask_pml] = alpha_max * ((pml_cells - dist_min[mask_pml]) / pml_cells) ** 2

# ------------------------ Геометрия ------------------------------------------
n_bg = np.float32(1.0)
n_wg = np.float32(3.5)
n_res = np.float32(3.5)

n_map = np.full((Nx, Ny, Nz), n_bg, dtype=np.float32)
contour_mask = np.zeros((Nx, Ny, Nz), dtype=np.uint8)

def broadcast_yz_mask_to_3d(mask_yz_2d: np.ndarray) -> np.ndarray:
    """
    Превращает маску формы (Ny, Nz) в форму (Nx, Ny, Nz),
    чтобы можно было безопасно индексировать n_map[mask3d].
    """
    return np.broadcast_to(mask_yz_2d[None, :, :], (Nx, Ny, Nz)).copy()


# Резонатор (шар)
res_cx = np.float32(float(Lx) / 2.0)
res_cy = np.float32(float(Ly) / 2.0)
res_cz = np.float32(float(Lz) / 2.0)

res_radius = 0.9 * (min(Lx, Ly, Lz) / 2.0 - pml_thickness_m)

# print(f"Радиус резонтора был выбран как")

dx2 = (x[:, None, None] - res_cx) ** 2
dy2 = (y[None, :, None] - res_cy) ** 2
dz2 = (z[None, None, :] - res_cz) ** 2
r2 = dx2 + dy2 + dz2

res_mask = r2 <= res_radius ** 2
n_map[res_mask] = n_res
contour_mask[res_mask] = 4

# Сферическая оболочка вокруг резонатора
strip_thickness = np.float32(0.05 * float(res_radius))
strip_mask = res_mask & (r2 >= (res_radius - strip_thickness) ** 2)
n_map[strip_mask] = n_res
contour_mask[strip_mask] = 5


# Верхний волновод (цилиндр вдоль X)
wg_t_radius = np.float32(0.25e-6)
wg_t_overlap = np.float32(0.1e-6)  # чтобы волновод немного заходил в резонатор

wg_t_cy = np.float32(res_cy + res_radius + wg_t_radius - wg_t_overlap)
wg_t_cz = np.float32(res_cz)

wg_t_mask_2d = ((y[:, None] - wg_t_cy) ** 2 + (z[None, :] - wg_t_cz) ** 2 <= wg_t_radius ** 2)
wg_t_mask = broadcast_yz_mask_to_3d(wg_t_mask_2d)

# n_map[wg_t_mask] = n_bg
n_map[wg_t_mask] = n_wg
contour_mask[wg_t_mask] = 2

# Нижний волновод (цилиндр вдоль X)
wg_b_radius = np.float32(0.25e-6)
wg_b_overlap = np.float32(0.1e-6)  # чтобы волновод немного заходил в резонатор

wg_b_cy = np.float32(res_cy - res_radius - wg_b_radius + wg_b_overlap)
wg_b_cz = np.float32(res_cz)

wg_b_mask_2d = ((y[:, None] - wg_b_cy) ** 2 + (z[None, :] - wg_b_cz) ** 2 <= wg_b_radius ** 2)
wg_b_mask = broadcast_yz_mask_to_3d(wg_b_mask_2d)

n_map[wg_b_mask] = n_wg
contour_mask[wg_b_mask] = 3

# PML
contour_mask[mask_pml] = 1

v2_map_base = (c0 / n_map) ** 2
v2_bg = np.float32((c0 / n_bg) ** 2)

# ------------------------ Начальное поле -------------------------------------
def init_field_directed_3d(freq, x0=None, A=0.7, theta_deg=0.0):
    """
    Создаёт начальный направленный пучок в нижнем волноводе.
    """
    if x0 is None:
        x0 = pml_thickness_m

    lam = c0 / np.float32(freq)
    print(f"длина волны для начального поля: {lam * 1e6:.2f} µm")
    
    k0 = np.float32(2.0 * np.pi / lam)
    theta = np.deg2rad(np.float32(theta_deg))
    kx = k0 * np.cos(theta)
    ky = k0 * np.sin(theta)

    Xc = x[:, None, None] - np.float32(x0)
    Yc = y[None, :, None] - wg_b_cy
    Zc = z[None, None, :] - wg_b_cz

    envelope = np.exp(-0.5 * ((Yc / np.float32(0.5e-6)) ** 2 + (Zc / np.float32(0.5e-6)) ** 2))
    phase = kx * Xc + ky * Yc
    E_init = np.float32(A) * envelope * np.cos(phase)

    source_window = x[:, None, None] < np.float32(0.45 * float(Lx))
    return np.where(source_window & wg_b_mask, E_init, np.float32(0.0)).astype(np.float32, copy=False)

# ------------------------ Шаг схемы ------------------------------------------
if HAVE_NUMBA:
    @njit(cache=True, fastmath=True, parallel=True)
    def step_fdtd(E, W, v2_map, alpha, dt, idx2, idy2, idz2, E_new, W_new):
        nx, ny, nz = E.shape
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    e = E[i, j, k]
                    lap = (
                        (E[i + 1, j, k] + E[i - 1, j, k] - 2.0 * e) * idx2
                        + (E[i, j + 1, k] + E[i, j - 1, k] - 2.0 * e) * idy2
                        + (E[i, j, k + 1] + E[i, j, k - 1] - 2.0 * e) * idz2
                    )
                    w_old = W[i, j, k]
                    w_new = w_old + dt * (v2_map[i, j, k] * lap - alpha[i, j, k] * w_old)
                    W_new[i, j, k] = w_new
                    E_new[i, j, k] = e + dt * w_new

    @njit(cache=True, fastmath=True)
    def energy_ratio(E, local_mask_f, overall_mask_f):
        local_energy = 0.0
        s_res = 0.0
        nx, ny, nz = E.shape
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    e2 = E[i, j, k] * E[i, j, k]
                    local_energy += e2 * local_mask_f[i, j, k]
                    s_res += e2 * overall_mask_f[i, j, k]
        return local_energy / (s_res + 1e-15)
else:
    def step_fdtd(E, W, v2_map, alpha, dt, idx2, idy2, idz2, E_new, W_new):
        E_new.fill(0.0)
        W_new.fill(0.0)
        lap = np.zeros_like(E)
        lap[1:-1, 1:-1, 1:-1] = (
            (E[2:, 1:-1, 1:-1] + E[:-2, 1:-1, 1:-1] - 2.0 * E[1:-1, 1:-1, 1:-1]) * idx2
            + (E[1:-1, 2:, 1:-1] + E[1:-1, :-2, 1:-1] - 2.0 * E[1:-1, 1:-1, 1:-1]) * idy2
            + (E[1:-1, 1:-1, 2:] + E[1:-1, 1:-1, :-2] - 2.0 * E[1:-1, 1:-1, 1:-1]) * idz2
        )
        W_new[:] = W + dt * (v2_map * lap - alpha * W)
        E_new[:] = E + dt * W_new

    def energy_ratio(E, strip_mask_f, res_mask_f):
        e2 = E * E
        return float(np.sum(e2 * strip_mask_f) / (np.sum(e2 * res_mask_f) + 1e-15))

# ------------------------ Визуализация ----------------------------------------
if USE_VISUALIZATION:
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    slice_z = Nz // 2
    slice_y = Ny // 2
    slice_x = Nx // 2

    im1 = axes[0].imshow(
        np.zeros((Ny, Nx), dtype=np.float32).T,
        origin="lower",
        extent=[0, float(Lx) * 1e6, 0, float(Ly) * 1e6],
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        interpolation="nearest",
        aspect="auto",
    )
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    axes[0].set_title(f"XY Slice (z={slice_z * float(dz) * 1e6:.2f} µm)")

    x_um = x * 1e6
    y_um = y * 1e6
    z_um = z * 1e6

    axes[0].contour(x_um, y_um, (contour_mask[:, :, slice_z] == 1).T, levels=[0.5], colors="black", linewidths=1.0)
    axes[0].contour(x_um, y_um, (contour_mask[:, :, slice_z] == 2).T, levels=[0.5], colors="blue", linewidths=1.8)
    axes[0].contour(x_um, y_um, (contour_mask[:, :, slice_z] == 3).T, levels=[0.5], colors="orange", linewidths=1.8)
    axes[0].contour(x_um, y_um, (contour_mask[:, :, slice_z] == 4).T, levels=[0.5], colors="red", linewidths=1.8)
    axes[0].contour(x_um, y_um, (contour_mask[:, :, slice_z] == 5).T, levels=[0.5], colors="green", linewidths=1.8)

    im2 = axes[1].imshow(
        np.zeros((Nz, Nx), dtype=np.float32).T,
        origin="lower",
        extent=[0, float(Lx) * 1e6, 0, float(Lz) * 1e6],
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        interpolation="nearest",
        aspect="auto",
    )
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("z (µm)")
    axes[1].set_title(f"XZ Slice (y={slice_y * float(dy) * 1e6:.2f} µm)")

    axes[1].contour(x_um, z_um, (contour_mask[:, slice_y, :] == 1).T, levels=[0.5], colors="black", linewidths=1.0)
    axes[1].contour(x_um, z_um, (contour_mask[:, slice_y, :] == 2).T, levels=[0.5], colors="blue", linewidths=1.8)
    axes[1].contour(x_um, z_um, (contour_mask[:, slice_y, :] == 3).T, levels=[0.5], colors="orange", linewidths=1.8)
    axes[1].contour(x_um, z_um, (contour_mask[:, slice_y, :] == 4).T, levels=[0.5], colors="red", linewidths=1.8)
    axes[1].contour(x_um, z_um, (contour_mask[:, slice_y, :] == 5).T, levels=[0.5], colors="green", linewidths=1.8)

    im3 = axes[2].imshow(
        np.zeros((Nz, Ny), dtype=np.float32).T,
        origin="lower",
        extent=[0, float(Ly) * 1e6, 0, float(Lz) * 1e6],
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        interpolation="nearest",
        aspect="auto",
    )
    axes[2].set_xlabel("y (µm)")
    axes[2].set_ylabel("z (µm)")
    axes[2].set_title(f"YZ Slice (x={slice_x * float(dx) * 1e6:.2f} µm)")

    axes[2].contour(y_um, z_um, (contour_mask[slice_x, :, :] == 1).T, levels=[0.5], colors="black", linewidths=1.0)
    axes[2].contour(y_um, z_um, (contour_mask[slice_x, :, :] == 2).T, levels=[0.5], colors="blue", linewidths=1.8)
    axes[2].contour(y_um, z_um, (contour_mask[slice_x, :, :] == 3).T, levels=[0.5], colors="orange", linewidths=1.8)
    axes[2].contour(y_um, z_um, (contour_mask[slice_x, :, :] == 4).T, levels=[0.5], colors="red", linewidths=1.8)
    axes[2].contour(y_um, z_um, (contour_mask[slice_x, :, :] == 5).T, levels=[0.5], colors="green", linewidths=1.8)

    plt.tight_layout()
    cbar = fig.colorbar(im1, ax=axes, orientation="vertical", fraction=0.02, pad=0.05)
    cbar.set_label("E amplitude")
    time_text = fig.text(0.5, 0.02, s="", ha="center", color="black", fontsize=12)
else:
    slice_z = Nz // 2
    slice_y = Ny // 2
    slice_x = Nx // 2
    fig = axes = im1 = im2 = im3 = time_text = None

# ------------------------ Параметры ------------------------------------------
# freqs = [i * float(f0) / 10 for i in range(10, 202, 20)] 
freqs = [i * float(f0) / 10 for i in range(10, 2011, 200)] # quick test

# test_freqs = freqs[-3:]   # для теста; замените на freqs, если нужен полный прогон
test_freqs = freqs[:]

nsteps_measure = 7500
# switch_step = 2500

strip_mask_f = strip_mask.astype(np.float32)
res_mask_f = res_mask.astype(np.float32)

# ------------------------ Симуляция ------------------------------------------
def run_simulation_3d(freq, i):
    E = np.ascontiguousarray(init_field_directed_3d(freq), dtype=np.float32)
    W = np.zeros_like(E, dtype=np.float32)

    E_new = np.empty_like(E)
    W_new = np.empty_like(W)

    v2_map = v2_map_base.copy()
    ratio = 0.0
    mem_gb = E.nbytes / 1e9

    ratios = np.empty(nsteps_measure // RATIO_INTERVAL + 1, dtype=np.float32)

    for n in range(nsteps_measure):
        t = n * float(dt)

        step_fdtd(E, W, v2_map, alpha, dt, idx2, idy2, idz2, E_new, W_new)

        # Нулевые граничные условия
        E_new[0, :, :] = 0.0
        E_new[-1, :, :] = 0.0
        E_new[:, 0, :] = 0.0
        E_new[:, -1, :] = 0.0
        E_new[:, :, 0] = 0.0
        E_new[:, :, -1] = 0.0

        W_new[0, :, :] = 0.0
        W_new[-1, :, :] = 0.0
        W_new[:, 0, :] = 0.0
        W_new[:, -1, :] = 0.0
        W_new[:, :, 0] = 0.0
        W_new[:, :, -1] = 0.0

        E, E_new = E_new, E
        W, W_new = W_new, W

        # Закрытие входа нижнего волновода
        # if n == switch_step:
        #     v2_map[wg_b_mask] = v2_bg

        # Отношение энергий
        if n % RATIO_INTERVAL == 0:
            ratio = float(energy_ratio(E, strip_mask_f, res_mask_f))
            # if ratio > max_ratio:
            #     max_ratio = ratio
            ratios[n // RATIO_INTERVAL] = ratio

        if n % PRINT_INTERVAL == 0:
            print(
                f"[i={i + 1}/{len(test_freqs)} f={freq / 1e12:.2f} THz] "
                f"step {n}/{nsteps_measure} ratio={ratio:.6e} mem={mem_gb:.2f} GB"
            )

        # Визуализация
        if USE_VISUALIZATION and (n % VIS_INTERVAL == 0):
            amp = float(np.max(np.abs(E)))
            amp = max(1e-12, amp)

            im1.set_data(E[:, :, slice_z].T)
            im2.set_data(E[:, slice_y, :].T)
            im3.set_data(E[slice_x, :, :].T)

            im1.set_clim(-0.9 * amp, 0.9 * amp)
            im2.set_clim(-0.9 * amp, 0.9 * amp)
            im3.set_clim(-0.9 * amp, 0.9 * amp)

            time_text.set_text(
                f"f={freq / 1e12:.2f} THz, t={t * 1e15:.1f} fs, step={n}, ratio={ratio:.3e}"
            )
            plt.pause(0.001)

    return ratios

# ------------------------ Главный запуск -------------------------------------
overall_start = time.time()
ratios = []

print("\n" + "=" * 60)
print("ЗАПУСК 3D FDTD СИМУЛЯЦИИ")
print("=" * 60)

for i, f in enumerate(test_freqs):
    print(f"\n=== Frequency = {f / 1e12:.2f} THz ===")
    # ratios.append(np.median(run_simulation_3d(f, i)))
    out = run_simulation_3d(f, i)
    ratios.append(np.median(out[len(out) // 4:]))

overall_end = time.time()
print(f"\n✅ Completed {len(test_freqs)} frequencies in {overall_end - overall_start:.2f} s")

# ------------------------ Финальный график -----------------------------------
plt.ioff()
plt.figure(figsize=(8, 5))

plt.plot(np.array(test_freqs) / 1e12, ratios, "o-", linewidth=2, markersize=8)
plt.xlabel("Frequency (THz)")
plt.ylabel("Max ratio E_strip / E_resonator")
plt.title("Dependence of energy ratio on frequency (3D)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

if headless:
    plt.savefig("ratio_vs_frequency.png", dpi=150)
    print("Сохранен файл ratio_vs_frequency.png")
else:
    plt.show(block=True)