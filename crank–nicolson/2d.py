# fdtd_wg_resonator_pml_viz_with_contours_cn.py
#
# Версия с консервативной схемой Кранка-Николсона (Crank-Nicolson) для волнового уравнения.
#  - Использует неявную схему (требуется решение СЛАУ на каждом шаге)
#  - Более устойчива и сохраняет энергию лучше явной схемы
#  - Использует scipy.sparse для ускорения работы с матрицами
#  - Для ускорения демонстрации количество шагов уменьшено до 3000 (вместо 10000),
#    а триггер изменения геометрии сдвинут на 1500 шаг.
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

# ----------------------- Сторонние библиотеки для линейной алгебры ---------------
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# -------------------------- Физические константы ---------------------------------
c0 = 299792458.0
lambda0 = 1.55e-6
f0 = c0 / lambda0
f0 = 1.9341449e14
lambda0 = c0 / f0

# -------------------------- Сетка ----------------------------------------------
Lx = 11.0e-6
Ly = 10.0e-6
dx = 0.03e-6
dy = dx
Nx = int(round(Lx / dx)) + 1
Ny = int(round(Ly / dy)) + 1

# Nx = 10
# Ny = 10


x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

n_min = 1.0
v_max = c0 / n_min
# Для неявной схемы CFL ограничение не строго необходимо для устойчивости, 
# но важно для точности дисперсии. Оставим прежним.
CFL = 0.35
CFL = 35
dt = CFL * dx / (v_max * math.sqrt(2.0))
idx2 = 1.0 / (dx * dx)

# -------------------------- PML --------------------------------------------------
pml_thickness_m = 0.05 * min(Lx, Ly)
pml_cells = int(round(pml_thickness_m / dx))
alpha_max = 0.8 / dt
ix = np.arange(Nx).reshape(-1, 1)
jy = np.arange(Ny).reshape(1, -1)
dist_x = np.minimum(ix, Nx - 1 - ix)
dist_y = np.minimum(jy, Ny - 1 - jy)
dist_min = np.minimum(dist_x, dist_y).astype(np.float64)
alpha = np.zeros_like(dist_min)
mask_pml = dist_min < pml_cells
alpha[mask_pml] = alpha_max * ((pml_cells - dist_min[mask_pml]) / pml_cells) ** 2

# -------------------------- Геометрия --------------------------------------------
n_bg = 1.0
n_wg = 5
n_res = 5
n_map = n_bg * np.ones((Nx, Ny), dtype=np.float64)
contour_mask = np.zeros((Nx, Ny), dtype=np.float64)

# Верхний волновод
wg_t_half_h = 0.2e-6
wg_t_cy = Ly - pml_thickness_m - 2 * wg_t_half_h
wg_t_mask = (np.abs(Y - wg_t_cy) <= wg_t_half_h)
n_map[wg_t_mask] = n_bg
contour_mask[wg_t_mask] = 2.0

# Нижний волновод
wg_b_half_h = 0.2e-6
wg_b_cy = pml_thickness_m + 2 * wg_b_half_h
wg_b_mask = (np.abs(Y - wg_b_cy) <= wg_b_half_h)
n_map[wg_b_mask] = n_wg
contour_mask[wg_b_mask] = 3.0

# Резонатор
res_cx = Lx / 2
res_cy = Ly / 2
overlap_b = 0.1e-6
res_radius = min(res_cx - pml_thickness_m, res_cy - (wg_b_cy + wg_b_half_h) + overlap_b)
res_mask = (X - res_cx)**2 + (Y - res_cy)**2 <= res_radius**2
n_map[res_mask] = n_res
contour_mask[res_mask] = 4.0

# Полоса вокруг резонатора
strip_thickness = 0.1 * res_radius
strip_mask = ((X - res_cx)**2 + (Y - res_cy)**2 >= (res_radius - strip_thickness)**2) & (res_mask)
n_map[strip_mask] = n_res
contour_mask[strip_mask] = 5.0

# PML
contour_mask[mask_pml] = 1.0

v_map = c0 / n_map
v2_map = v_map ** 2

# -------------------------- Построение разреженной матрицы Лапласиана -----------
def build_laplacian_matrix(Nx, Ny, dx):
    """Создает разреженную матрицу оператора Лапласа с Дирихле граничными условиями."""
    N = Nx * Ny
    main_diag = -4.0 / (dx * dx) * np.ones(N)
    off_diag_x = 1.0 / (dx * dx) * np.ones(N)
    off_diag_y = 1.0 / (dx * dx) * np.ones(N)
    
    # Исключаем связи на границах для соблюдения условий Дирихле (нулевое поле)
    # Границы по X (ось 0): строки 0 и Nx-1 в каждом столбце Y
    # В линейном индексе (C-order, но у нас indexing='ij' -> X меняется медленнее? 
    # В numpy meshgrid indexing='ij': X[i,j] = x[i]. shape (Nx, Ny).
    # ravel() по умолчанию 'C': последний индекс меняется быстрее. 
    # Значит индекс k = i * Ny + j.
    # Соседи по Y: k+1, k-1. Соседи по X: k+Ny, k-Ny.
    
    # Обнуляем связи на границах сетки
    # Границы j=0 и j=Ny-1 (по оси Y)
    for i in range(Nx):
        idx_start = i * Ny
        off_diag_y[idx_start] = 0       # Левая граница по Y
        off_diag_y[idx_start + Ny - 1] = 0 # Правая граница по Y (связь с следующим элементом отсутствует)
        # Также нужно убрать связь "входящую" в границу, но в диагональной матрице смещения 
        # это обрабатывается автоматически, если мы не создаем связь из граничного узла.
        # Однако, для узла на границе (где E=0), мы можем просто не включать их в систему 
        # или оставить с большим коэффициентом. Проще: оставить узлы, но обнулить связи с "внешним миром".
        # В схеме Кранка-Николсона с Дирихле, мы просто не обновляем граничные узлы (они 0).
        # Поэтому матрица должна быть для внутренних узлов, или граничные узлы должны иметь 
        # единичную строку (E=0). 
        # Для простоты оставим все узлы, но обеспечим, чтобы граничные не влияли на внутренние 
        # и сами оставались 0.
        
    # Границы i=0 и i=Nx-1 (по оси X)
    # Индексы: 0..Ny-1 и (Nx-1)*Ny .. N-1
    for j in range(Ny):
        idx_top = j
        idx_bot = (Nx - 1) * Ny + j
        off_diag_x[idx_top] = 0
        off_diag_x[idx_bot] = 0
        
    # Диагонали
    offsets = [0, 1, -1, Ny, -Ny]
    diags = [main_diag, off_diag_y, off_diag_y, off_diag_x, off_diag_x]
    
    L_mat = sp.diags(diags, offsets, shape=(N, N), format='csc')

    # print(L_mat.toarray().shape)  # Печать части матрицы для проверки
    # print(L_mat.toarray()[:10, :10])  # Печать части матрицы для проверки

    return L_mat

L_mat = build_laplacian_matrix(Nx, Ny, dx)

# -------------------------- Инициализация поля ----------------------------------
def init_field_directed(X, Y, freq, x0=pml_thickness_m, A=0.7, theta_deg=0.0):
    """Создаёт начальный направленный пучок в нижнем волноводе."""
    lam = c0 / freq
    k0 = 2 * np.pi / lam
    theta = np.deg2rad(theta_deg)
    kx = k0 * np.cos(theta)
    ky = k0 * np.sin(theta)
    Xc = X - x0
    Yc = Y - wg_b_cy
    envelope = np.exp(-0.5 * (Yc / (0.5e-6))**2)
    phase = kx * Xc + ky * Yc
    E_init = A * envelope * np.cos(phase)
    return np.where(X < 0.9 * Lx / 2, E_init * wg_b_mask, 0)

# -------------------------- Источник --------------------------------------------
src_x_cols = [int(round(1.0e-6 / dx)), int(round(1.0e-6 / dx)) + 1]
src_y0 = wg_b_cy
src_sigma = 0.7e-6
src_mask = np.exp(-0.5 * ((Y - src_y0) / src_sigma)**2)
src_amplitude = 0.5
def src_time(t): return 0.0

# -------------------------- Визуализация ----------------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(12, 9))
im = ax.imshow(np.zeros_like(X).T, origin='lower',
               extent=[0, Lx*1e6, 0, Ly*1e6],
               cmap='RdBu_r', vmin=-0.5, vmax=0.5,
               interpolation='bilinear')
ax.set_xlabel('x (µm)')
ax.set_ylabel('y (µm)')
ax.set_title('E_z field (Crank-Nicolson Implicit Scheme)')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('E amplitude')

# === Добавляем контуры областей ===
ax.contour(X.T * 1e6, Y.T * 1e6, (contour_mask == 1).T,
           levels=[0.5], colors='black', linewidths=1.2, label='PML')
ax.contour(X.T * 1e6, Y.T * 1e6, (contour_mask == 2).T,
           levels=[0.5], colors='blue', linewidths=2.5, label='Top WG')
ax.contour(X.T * 1e6, Y.T * 1e6, (contour_mask == 3).T,
           levels=[0.5], colors='orange', linewidths=2.5, label='Bottom WG')
ax.contour(X.T * 1e6, Y.T * 1e6, (contour_mask == 4).T,
           levels=[0.5], colors='red', linewidths=2.5, label='Resonator')
ax.contour(X.T * 1e6, Y.T * 1e6, (contour_mask == 5).T,
           levels=[0.5], colors='green', linewidths=2.5, label='Strip')

time_text = ax.text(0.01, 0.97, s='', transform=ax.transAxes, color='black', fontsize=12,)

# -------------------------- Параметры ------------------------------------------
# Уменьшено количество шагов для демонстрации неявной схемы (она медленнее явной)
freqs = [i * f0 / 10 for i in range(10, 105, 5)]
nsteps_measure = 3000 
vis_interval = 10
print_interval = 100
geometry_change_step = 1500  # Сдвинуто для демонстрации в рамках 3000 шагов

# -------------------------- Основная функция симуляции --------------------------
def run_simulation(freq, i, v2_map_init):
    """Запуск симуляции с схемой Кранка-Николсона."""
    
    # 1. Инициализация полей
    E = init_field_directed(X, Y, freq)
    W = np.zeros_like(E) # W ~ dE/dt
    
    # Маски для измерений
    top_mask = wg_t_mask.astype(float)
    res_mask_f = res_mask.astype(float)
    strip_mask_f = strip_mask.astype(float)
    
    # Векторизация для работы с разреженными матрицами
    # indexing='ij' + ravel() (C-order) -> k = i*Ny + j
    # Нужно быть внимательным: L_mat построена предполагая эту структуру.
    
    # 2. Построение матрицы системы для Кранка-Николсона
    # Система выведена для первого порядка по времени:
    # dE/dt = W
    # dW/dt = v^2 Laplacian E - alpha W
    # Схема CN приводит к уравнению для W^{n+1}:
    # (I + dt/2*diag(alpha) - dt^2/4*diag(v^2)*Lap) W^{n+1} = RHS
    # Где RHS зависит от E^n, W^n.
    
    N = Nx * Ny
    I_mat = sp.eye(N, format='csc')
    
    # Диагональные матрицы коэффициентов
    diag_alpha = sp.diags(alpha.ravel(), 0, format='csc')
    diag_v2 = sp.diags(v2_map_init.ravel(), 0, format='csc') # TODO: понять зачем создавать матрицу, а не просто домножить на число при формировании M в цикле. Ответ: v^2 может меняться при изменении геометрии, нужно обновлять матрицу M.
    
    # Оператор в ЛЧ (LHS): M = I + 0.5*dt*alpha - 0.25*dt^2*v^2*Laplacian
    # Обратите внимание: L_mat уже содержит знак Лапласиана (отрицательно определенный)
    # В уравнении: ... - 0.25 * dt^2 * v^2 * (Laplacian E)
    # Значит в матрице M коэффициент при L_mat будет -0.25 * dt^2 * v^2
    # Но так как L_mat отрицательный, -L_mat положительный.
    
    M = I_mat + 0.5 * dt * diag_alpha - 0.25 * dt**2 * (diag_v2 @ L_mat)
    
    # Факторизация матрицы (LU decomposition) для быстрого решения на каждом шаге
    # Это самая тяжелая операция, делается один раз (или при изменении геометрии)
    try:
        solver = spla.factorized(M.tocsc())
    except Exception as e:
        print(f"Error factorizing matrix: {e}")
        return np.zeros(nsteps_measure)

    output = np.zeros(nsteps_measure)
    current_v2_map = v2_map_init.copy()
    matrix_updated = False

    # Граничные индексы (для принудительного обнуления после решения, если нужно)
    # В нашей матрице мы обнулили связи, но значения на границе могут дрейфовать численно.
    # Лучше явно обнулять.
    boundary_mask = np.zeros((Nx, Ny), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_indices = np.where(boundary_mask.ravel())[0]

    for n in range(nsteps_measure):
        t = n * dt
        
        # 1. Изменение геометрии (если наступил шаг)
        if n == geometry_change_step and not matrix_updated:
            # Обновляем карту скоростей (закрытие волновода)
            n_map[wg_b_mask] = n_bg
            v_map = c0 / n_map
            current_v2_map = v_map ** 2
            
            # Пересобираем матрицу
            diag_v2_new = sp.diags(current_v2_map.ravel(), 0, format='csc')
            M_new = I_mat + 0.5 * dt * diag_alpha - 0.25 * dt**2 * (diag_v2_new @ L_mat)
            solver = spla.factorized(M_new.tocsc())
            matrix_updated = True
            print(f"[Step {n}] Geometry updated, matrix refactored.")

        # 2. Формирование правой части (RHS) для уравнения на W
        # RHS = (I - 0.5*dt*alpha + 0.25*dt^2*v^2*Lap) W^n + dt * v^2 * Lap E^n + Source
        # Упрощенно: RHS = W^n - 0.5*dt*alpha*W^n + 0.25*dt^2*v^2*Lap(W^n) + dt*v^2*Lap(E^n) + Source
        
        E_flat = E.ravel()
        W_flat = W.ravel()
        
        # Лапласиан от текущих полей (разреженное умножение)
        LapE = L_mat @ E_flat
        LapW = L_mat @ W_flat
        
        # Коэффициенты
        v2_flat = current_v2_map.ravel()
        alpha_flat = alpha.ravel()
        
        # RHS calculation
        # Terms involving W^n
        term_W = W_flat - 0.5 * dt * alpha_flat * W_flat + 0.25 * dt**2 * v2_flat * LapW
        # Terms involving E^n
        term_E = dt * v2_flat * LapE
        
        rhs = term_W + term_E
        
        # Источник (добавляем в RHS для W, так как источник влияет на производную поля)
        # inj = src_time(t) * src_amplitude
        # if inj != 0.0:
        #     for col in src_x_cols:
        #         if 0 <= col < Nx:
        #             # Индексы в плоском массиве для столбца col
        #             idx_start = col * Ny
        #             idx_end = (col + 1) * Ny
        #             # Профиль источника по Y
        #             src_vec = inj * src_mask[col, :]
        #             rhs[idx_start:idx_end] += src_vec

        # 3. Решение СЛАУ
        try:
            W_new_flat = solver(rhs)
        except Exception:
            # Fallback if solver fails (rare)
            W_new_flat = rhs 

        # 4. Обновление E
        # E^{n+1} = E^n + 0.5 * dt * (W^{n+1} + W^n)
        E_new_flat = E_flat + 0.5 * dt * (W_new_flat + W_flat)
        
        # 5. Граничные условия (Дирихле: E=0, W=0 на краях)
        E_new_flat[boundary_indices] = 0.0
        W_new_flat[boundary_indices] = 0.0
        
        # Reshape back to 2D
        E = E_new_flat.reshape((Nx, Ny))
        W = W_new_flat.reshape((Nx, Ny))

        # Измерения
        ratio = np.sum((E**2) * strip_mask_f) / (np.sum((E**2) * res_mask_f) + 1e-12)
        output[n] = ratio

        # Вывод в консоль
        if n % print_interval == 0:
            print(f"[i={i}/{len(freqs)} f={freq/1e12:.1f} THz] step {n}/{nsteps_measure} ratio={ratio:.6f}")

        # Обновление анимации
        if n % vis_interval == 0:
            im.set_data(E.T)
            amp = max(1e-12, np.max(np.abs(E)))
            im.set_clim(-0.9*amp, 0.9*amp)
            time_text.set_text(f"f={freq/1e12:.1f} THz, t={t*1e15:.1f} fs, step={n} (CN)")
            plt.pause(0.001)

    return output

# -------------------------- Основной цикл по частотам --------------------------
max_ratios = []
overall_start = time.time()

# Берем только несколько частот для демонстрации, так как CN схема медленная
# Если хотите полный скан, уберите [:5] но будьте готовы ждать
demo_freqs = freqs[-5:] 
print(f"Running Crank-Nicolson simulation for {len(demo_freqs)} frequencies...")
print("Note: Implicit scheme is computationally heavy. Steps reduced for demo.")

for i, f in enumerate(demo_freqs):
    print(f"\n === Frequency = {f/1e12:.2f} THz ===")
    # Передаем начальную карту v2, внутри функции она может обновиться
    output = run_simulation(f, i, v2_map)
    # Берем среднее после установления режима (примерно со второй половины)
    start_idx = int(nsteps_measure * 0.5)
    max_ratios.append(output[start_idx:].mean())  
    
overall_end = time.time()

print(f"\n✅ Completed {len(demo_freqs)} frequencies in {overall_end - overall_start:.2f} s")

# -------------------------- График зависимости ---------------------------------
plt.ioff()
plt.figure(figsize=(8,5))
plt.plot(np.array(demo_freqs)/1e12, max_ratios, 'o-', linewidth=2, color='darkgreen')
plt.xlabel('Frequency (THz)')
plt.ylabel('Avg ratio E_strip / E_resonator')
plt.title('Energy Ratio vs Frequency (Crank-Nicolson Scheme)')
plt.grid(True)
plt.show(block=True)