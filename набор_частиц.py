import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.interpolate import interp1d
import time

# Параметры
lambda_wave = 0.6328  
n_medium = 1.33
n_particle = 1.05
m = n_particle / n_medium
k = 2 * np.pi / lambda_wave
""" a0 - толщина в центре a1 - степень вонутости a2 - коррекция формы """
# Дискретизация для DDA
points_per_lambda = 4
dipole_size = lambda_wave / points_per_lambda

R_min = 2.0
R_max = 7.0
num_R = 40
R_vals = np.linspace(R_min, R_max, num_R)

theta_max = np.radians(25)
theta_vals = np.linspace(0.001, theta_max, 300)



# Функции распределения
def distribution_normal(R):
    return np.exp(-20 * (R - 3.5)**2)

def distribution_bimodal(R):
    return (2/3) * np.exp(-20 * (R - 3.5)**2) + (1/3) * np.exp(-20 * (R - 4.0)**2)

def normalize_distribution(R, dist_func):
    weights = dist_func(R)
    integral = np.trapz(R**2 * weights, R)
    return weights / integral

omega_normal = normalize_distribution(R_vals, distribution_normal)
omega_bimodal = normalize_distribution(R_vals, distribution_bimodal)


def biconcave_profile(r, D, a0=0.15, a1=1.2, a2=0.1):
    r_norm = 2 * r / D
    r_norm = np.clip(r_norm, 0, 0.999)
    sqrt_term = np.sqrt(1 - r_norm**2)
    poly_term = a0 + a1 * r_norm**2 + a2 * r_norm**4
    return D * sqrt_term * poly_term

def create_dipole_grid(shape, radius, a0=0.15, a1=1.2, a2=0.1):
    dipoles = []
    size = int(2 * radius / dipole_size) + 1
    D = 2 * radius
    
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            for k in range(-size, size+1):
                x = i * dipole_size
                y = j * dipole_size
                z = k * dipole_size
                
                if shape == 'sphere':
                    if x**2 + y**2 + z**2 <= radius**2:
                        dipoles.append([x, y, z])
                
                elif shape == 'ellipsoid':
                    a = radius
                    b = radius / 1.5
                    c = radius / 1.5
                    if (x/a)**2 + (y/b)**2 + (z/c)**2 <= 1:
                        dipoles.append([x, y, z])
                
                elif shape == 'biconcave':
                    r = np.sqrt(x**2 + y**2)
                    if r <= radius:
                        z_surface = biconcave_profile(r, D, a0, a1, a2)
                        if abs(z) <= z_surface:
                            dipoles.append([x, y, z])
    
    return np.array(dipoles)

def adda_dda(theta_vals, dipoles):
    if len(dipoles) == 0:
        return np.zeros_like(theta_vals)
    
    wavelength_in_medium = lambda_wave / n_medium
    k_medium = 2 * np.pi / wavelength_in_medium
    alpha = dipole_size**3 * (m**2 - 1) / (m**2 + 2)
    
    I = []
    N = len(dipoles)
    
    for theta in theta_vals:
        if theta == 0:
            I.append(1.0)
            continue
        
        k_inc = np.array([0, 0, k_medium]) #падающий вектор волны 
        k_sca = np.array([k_medium * np.sin(theta), 0, k_medium * np.cos(theta)]) #рассеяный вектор волны 
        q = k_sca - k_inc
        
        phases = np.exp(1j * np.dot(dipoles, q))
        amplitude = np.sum(phases)
        intensity = np.abs(alpha * amplitude)**2
        intensity /= N**2
        I.append(intensity)
    
    I = np.array(I)
    if np.max(I) > 0:
        I = I / np.max(I)
    
    return I

def sphere_intensity(theta_vals, radius):
    I = []
    for theta in theta_vals:
        if theta == 0:
            I.append(1.0)
        else:
            x = k * radius * theta
            j1 = jv(1, x)
            I.append((2 * j1 / x)**2)
    return np.array(I)

def ellipse_intensity(theta_vals, a, b):
    I = []
    for theta in theta_vals:
        if theta == 0:
            I.append(1.0)
        else:
            x = k * np.sqrt((a**2 + b**2)/2) * theta
            j1 = jv(1, x)
            I.append((2 * j1 / x)**2)
    return np.array(I)


sphere_matrix = np.zeros((len(theta_vals), num_R))
for i, R in enumerate(R_vals):
    sphere_matrix[:, i] = sphere_intensity(theta_vals, R)

ellipse_matrix = np.zeros((len(theta_vals), num_R))
for i, R in enumerate(R_vals):
    ellipse_matrix[:, i] = ellipse_intensity(theta_vals, R, R/1.5)

disk_matrix = np.zeros((len(theta_vals), num_R))
sample_indices = np.linspace(0, num_R-1, 8, dtype=int)

print("Вычисление DDA для диска:")
for idx in sample_indices:
    R = R_vals[idx]
    print(f"  Радиус R={R:.2f} мкм ")
    dipoles = create_dipole_grid('biconcave', R)
    disk_matrix[:, idx] = adda_dda(theta_vals, dipoles)

for j in range(len(theta_vals)):
    f = interp1d(R_vals[sample_indices], disk_matrix[j, sample_indices], 
                 kind='linear', fill_value='extrapolate')
    disk_matrix[j, :] = f(R_vals)

def find_extrema(theta_vals, intensity):
    min_idx = None
    for i in range(5, len(intensity)-5):
        if intensity[i] < intensity[i-1] and intensity[i] < intensity[i+1]:
            if intensity[i] < 0.5:
                min_idx = i
                break
    
    max_idx = None
    for i in range(5, len(intensity)-5):
        if intensity[i] > intensity[i-1] and intensity[i] > intensity[i+1]:
            if theta_vals[i] > 0.02:
                max_idx = i
                break
    
    theta_min = theta_vals[min_idx] if min_idx else None
    I_min = intensity[min_idx] if min_idx else None
    theta_max = theta_vals[max_idx] if max_idx else None
    I_max = intensity[max_idx] if max_idx else None
    
    return theta_min, I_min, theta_max, I_max

def compute_ensemble_intensity(intensity_matrix, distribution):
    result = np.zeros(len(theta_vals))
    for j in range(len(theta_vals)):
        result[j] = np.trapz(intensity_matrix[j, :] * distribution, R_vals)
    
    if np.max(result) > 0:
        result = result / np.max(result)
    
    return result



results = {}
shapes = ['sphere', 'ellipsoid', 'biconcave']
shape_names = {'sphere': 'Сфера', 'ellipsoid': 'Эллипсоид', 'biconcave': 'Диск'}
dist_names = {'normal': 'Нормальное', 'bimodal': 'Бимодальное'}
matrices = {'sphere': sphere_matrix, 'ellipsoid': ellipse_matrix, 'biconcave': disk_matrix}

for shape in shapes:
    for dist_name, omega in [('normal', omega_normal), ('bimodal', omega_bimodal)]:
        key = f"{shape_names[shape]} ({dist_names[dist_name]})"
        print(f"  {key}...")
        
        I_ensemble = compute_ensemble_intensity(matrices[shape], omega)
        theta_min, I_min, theta_max, I_max = find_extrema(theta_vals, I_ensemble)
        
        results[key] = {
            'intensity': I_ensemble,
            'theta_min': theta_min,
            'I_min': I_min,
            'theta_max': theta_max,
            'I_max': I_max
        }
        
        print(f"teta_min = {theta_min:.4f} рад, I_min = {I_min:.6f}")
        print(f"teta_max = {theta_max:.4f} рад, I_max = {I_max:.6f}")

# Построение графиков
fig = plt.figure(figsize=(20, 14))
fig.subplots_adjust(hspace=0.3, wspace=0.25)

# Создаем сетку с отступами
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

colors = {
    'Сфера (Нормальное)': 'blue',
    'Сфера (Бимодальное)': 'cyan',
    'Эллипсоид (Нормальное)': 'red',
    'Эллипсоид (Бимодальное)': 'orange',
    'Диск (Нормальное)': 'green',
    'Диск (Бимодальное)': 'lime'
}

# 1. Дифракционная картина
ax1 = fig.add_subplot(gs[0, 0])
for name, data in results.items():
    ax1.plot(theta_vals, data['intensity'], 
            color=colors[name], linewidth=2, label=name, alpha=0.8)
    
    if data['theta_min']:
        ax1.scatter(data['theta_min'], data['I_min'], 
                  color=colors[name], s=120, marker='v', 
                  edgecolors='black', zorder=5, label='_nolegend_')
    
    if data['theta_max']:
        ax1.scatter(data['theta_max'], data['I_max'], 
                  color=colors[name], s=120, marker='^', 
                  edgecolors='black', zorder=5, label='_nolegend_')

ax1.set_xlabel('Угол teta (рад)', fontsize=12)
ax1.set_ylabel('Интенсивность I(teta)', fontsize=12)
ax1.set_title('Дифракционные картины', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=8, ncol=1, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 0.3])
ax1.set_ylim([0, 1.1])

# 2. Логарифмическая шкала
ax2 = fig.add_subplot(gs[0, 1])
for name, data in results.items():
    ax2.semilogy(theta_vals, data['intensity'] + 1e-10,
                color=colors[name], linewidth=2, label=name, alpha=0.8)
    
    if data['theta_min']:
        ax2.scatter(data['theta_min'], data['I_min'] + 1e-10,
                  color=colors[name], s=120, marker='v', 
                  edgecolors='black', zorder=5, label='_nolegend_')
    
    if data['theta_max']:
        ax2.scatter(data['theta_max'], data['I_max'] + 1e-10,
                  color=colors[name], s=120, marker='^', 
                  edgecolors='black', zorder=5, label='_nolegend_')

ax2.set_xlabel('Угол teta (рад)', fontsize=12)
ax2.set_ylabel('Интенсивность (лог. шкала)', fontsize=12)
ax2.set_title('Логарифмическая шкала', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8, ncol=1, framealpha=0.9)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim([0, 0.3])
ax2.set_ylim([1e-6, 1])

# 3. Распределения по размерам
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(R_vals, omega_normal, 'b-', linewidth=2, label='Нормальное')
ax3.plot(R_vals, omega_bimodal, 'r-', linewidth=2, label='Бимодальное')
ax3.set_xlabel('Радиус R (мкм)', fontsize=12)
ax3.set_ylabel('Плотность w(R)', fontsize=12)
ax3.set_title('Распределения частиц', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([R_min, R_max])

# 4. ТАБЛИЦА С РЕЗУЛЬТАТАМИ 
ax4 = fig.add_subplot(gs[1, 0])
ax4.axis('off')

# Собираем данные для таблицы
table_data = []
headers = ['Форма', 'Распред.', '0_min', 'I_min', '0_max', 'I_max']
table_data.append(headers)

for name, data in results.items():
    parts = name.split(' (')
    shape = parts[0]
    dist = parts[1].rstrip(')')
    table_data.append([
        shape,
        dist,
        f'{data["theta_min"]:.4f}',
        f'{data["I_min"]:.6f}',
        f'{data["theta_max"]:.4f}',
        f'{data["I_max"]:.6f}'
    ])



# 5. График зависимости интенсивности от ω(R)
ax5 = fig.add_subplot(gs[1, 1])

# Выбираем фиксированный угол (первый минимум для нормального распределения)
theta_fixed_idx = np.argmin(np.abs(theta_vals - 0.09))  # около первого минимума

# Для разных форм показываем вклад каждого радиуса
for shape_name, matrix, style in [
    ('Сфера', sphere_matrix, '-'), 
    ('Эллипсоид', ellipse_matrix, '--'), 
    ('Диск', disk_matrix, ':')
]:
    I_at_theta = matrix[theta_fixed_idx, :]
    contribution_normal = I_at_theta * omega_normal
    contribution_bimodal = I_at_theta * omega_bimodal
    
    ax5.plot(R_vals, contribution_normal, color='blue', linestyle=style, linewidth=1.8, 
            label=f'{shape_name} (норм.)', alpha=0.8)
    ax5.plot(R_vals, contribution_bimodal, color='red', linestyle=style, linewidth=1.8, 
            label=f'{shape_name} (бимод.)', alpha=0.8)

ax5.set_xlabel('Радиус R (мкм)', fontsize=12)
ax5.set_ylabel(f'Вклад I·w(R) при teta={theta_vals[theta_fixed_idx]:.3f} рад', fontsize=11)
ax5.set_title('Зависимость интенсивности от w(R)', fontsize=14, fontweight='bold')
ax5.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
ax5.grid(True, alpha=0.3)
ax5.set_xlim([R_min, R_max])

# 6. Информационная панель
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

info_text = (
    "ПАРАМЕТРЫ РАСЧЕТА:\n"
    f"длина волны = {lambda_wave} мкм\n"
    f"n_среды = {n_medium}\n"
    f"n_частицы = {n_particle}\n"
    f"R ∈ [{R_min}, {R_max}] мкм\n"
    f"точек по R: {num_R}\n"
    f"точек по 0: {len(theta_vals)}\n"
    f"0фикс = {theta_vals[theta_fixed_idx]:.3f} рад\n\n"

)

ax6.text(0.05, 0.5, info_text, transform=ax6.transAxes, fontsize=11, 
        fontfamily='monospace', verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='#f9f9f9', alpha=0.9, edgecolor='gray'))

plt.suptitle('ДИФРАКЦИЯ НА АНСАМБЛЕ ЧАСТИЦ', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Таблица 
print(f"{'Форма':<15} {'Распределение':<15} {'0_min (рад)':<12} {'I_min':<14} {'0_max (рад)':<12} {'I_max':<14}")
print("-"*110)

for name, data in results.items():
    parts = name.split(' (')
    shape = parts[0]
    dist = parts[1].rstrip(')')
    print(f"{shape:<15} {dist:<15} {data['theta_min']:.4f}       {data['I_min']:.8f}    {data['theta_max']:.4f}       {data['I_max']:.8f}")