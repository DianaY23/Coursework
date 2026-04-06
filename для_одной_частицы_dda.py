import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.signal import find_peaks
import time

lambda_wave = 0.6328
n_medium = 1.33
n_particle = 1.05
m = n_particle / n_medium
R_typical = 3.75
k = 2 * np.pi / lambda_wave

# Дискретизация
points_per_lambda = 8
dipole_size = lambda_wave / points_per_lambda
print(f"Размер диполя: {dipole_size:.3f} мкм")
print(f"Точек на длину волны: {points_per_lambda}")

def create_sphere(radius):
    dipoles = []
    size = int(2 * radius / dipole_size) + 1
    
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            for k in range(-size, size+1):
                x = i * dipole_size
                y = j * dipole_size
                z = k * dipole_size
                
                if x**2 + y**2 + z**2 <= radius**2:
                    dipoles.append([x, y, z])
    
    return np.array(dipoles)

def create_flat_disk(radius, height_factor):
    dipoles = []
    height = radius * height_factor
    size = int(2 * radius / dipole_size) + 1
    
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            for k in range(-size, size+1):
                x = i * dipole_size
                y = j * dipole_size
                z = k * dipole_size
                
                # Радиус в плоскости XY
                r = np.sqrt(x**2 + y**2)
                
                if r <= radius and abs(z) <= height/2:
                    dipoles.append([x, y, z])
    
    return np.array(dipoles)

def create_biconcave_disk(radius, outer_height_factor, inner_height_factor):
    """
    Создает двояковогнутый диск - толщина уменьшается к центру
    outer_height_factor: высота на краю
    inner_height_factor: высота в центре (должна быть меньше outer_height_factor)
    """
    dipoles = []
    outer_height = radius * outer_height_factor  # высота на краю
    inner_height = radius * inner_height_factor  # высота в центре (меньше)
    
    size = int(2 * radius / dipole_size) + 1
    
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            for k in range(-size, size+1):
                x = i * dipole_size
                y = j * dipole_size
                z = k * dipole_size
                
                r = np.sqrt(x**2 + y**2)
                if r <= radius:
                    r_norm = r / radius
                    # Профиль: на краю максимальная толщина, в центре минимальная
                    # Формула: толщина = внешняя_высота - (внешняя_высота - внутренняя_высота) * (1 - r_norm^2)^2
                    thickness = outer_height - (outer_height - inner_height) * (1 - r_norm**2)**2
                    z_surface = thickness / 2
                    
                    if abs(z) <= z_surface:
                        dipoles.append([x, y, z])
    
    return np.array(dipoles)

def create_rotated_ellipsoid(radius, angle_deg):
    dipoles = []
    angle_rad = np.radians(angle_deg)
    
    a = radius  
    b = radius / 2.5  
    c = radius / 2.5 
    
    size = int(2 * radius / dipole_size) + 1
    
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            for k in range(-size, size+1):
                x = i * dipole_size
                y = j * dipole_size
                z = k * dipole_size
                
                x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
                y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
                
                if (x_rot/a)**2 + (y_rot/b)**2 + (z/c)**2 <= 1:
                    dipoles.append([x, y, z])
    
    return np.array(dipoles)

def adda_dda(theta_vals, dipoles, plane='xz'):
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
        
        k_inc = np.array([0, 0, k_medium])
        
        if plane == 'xz':
            k_sca = np.array([k_medium * np.sin(theta), 0, k_medium * np.cos(theta)])
        else:
            k_sca = np.array([0, k_medium * np.sin(theta), k_medium * np.cos(theta)])
            
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

def find_extrema(theta_vals, intensity):
    min_peaks = find_peaks(-intensity, distance=10)[0]
    max_peaks = find_peaks(intensity, distance=10)[0]
    
    first_min = None
    first_min_val = None
    first_max = None
    first_max_val = None

    for idx in min_peaks:
        if idx > 5 and idx < len(intensity)-5:
            if intensity[idx] < 0.5:
                first_min = theta_vals[idx]
                first_min_val = intensity[idx]
                break
    
    for idx in max_peaks:
        if theta_vals[idx] > 0.02:
            first_max = theta_vals[idx]
            first_max_val = intensity[idx]
            break
    
    return first_min, first_min_val, first_max, first_max_val

def plot_3d_models_with_angle(angle):
    """Функция для 3D визуализации с заданным углом - все объекты с сеткой"""
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Сфера - полная поверхность с сеткой
    ax1 = fig.add_subplot(3, 4, 1, projection='3d')
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    u, v = np.meshgrid(u, v)
    
    x = R_typical * np.sin(v) * np.cos(u)
    y = R_typical * np.sin(v) * np.sin(u)
    z = R_typical * np.cos(v)
    
    ax1.plot_surface(x, y, z, color='cyan', alpha=0.4, edgecolor='blue', linewidth=0.5)
    ax1.set_title('Сфера', fontsize=10, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-R_typical-1, R_typical+1])
    ax1.set_ylim([-R_typical-1, R_typical+1])
    ax1.set_zlim([-R_typical-1, R_typical+1])
    ax1.view_init(elev=25, azim=45)
    
    # 2. Плоские диски - полные цилиндры с сеткой
    height_factors = [1, 0.5, 0.125]
    titles = ['h = R', 'h = R/2', 'h = R/8']
    colors_disk = ['blue', 'royalblue', 'navy']
    
    for idx, (h_factor, title, color) in enumerate(zip(height_factors, titles, colors_disk)):
        ax = fig.add_subplot(3, 4, idx+2, projection='3d')
        
        height = R_typical * h_factor
        
        # Боковая поверхность
        theta_side = np.linspace(0, 2*np.pi, 50)
        z_side = np.linspace(-height/2, height/2, 30)
        Theta_side, Z_side = np.meshgrid(theta_side, z_side)
        
        X_side = R_typical * np.cos(Theta_side)
        Y_side = R_typical * np.sin(Theta_side)
        
        # Верхняя и нижняя крышки
        theta_top = np.linspace(0, 2*np.pi, 50)
        r_top = np.linspace(0, R_typical, 30)
        R_top, Theta_top = np.meshgrid(r_top, theta_top)
        X_top = R_top * np.cos(Theta_top)
        Y_top = R_top * np.sin(Theta_top)
        Z_top = np.ones_like(X_top) * height/2
        Z_bottom = np.ones_like(X_top) * -height/2
        
        # Рисуем все поверхности с сеткой
        ax.plot_surface(X_side, Y_side, Z_side, color=color, alpha=0.3, edgecolor='darkblue', linewidth=0.5)
        ax.plot_surface(X_top, Y_top, Z_top, color=color, alpha=0.3, edgecolor='darkblue', linewidth=0.5)
        ax.plot_surface(X_top, Y_top, Z_bottom, color=color, alpha=0.3, edgecolor='darkblue', linewidth=0.5)
        
        ax.set_title(f'Плоский диск, {title}', fontsize=10, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-R_typical-1, R_typical+1])
        ax.set_ylim([-R_typical-1, R_typical+1])
        ax.set_zlim([-R_typical-1, R_typical+1])
        ax.view_init(elev=25, azim=45)
    
    # 3. Двояковогнутые диски - настоящая вогнутость (тоньше в центре) с сеткой
    biconcave_configs = [
        (1, 0.25, 'h_out=R, h_in=R/4', 'limegreen'),
        (1, 0.125, 'h_out=R, h_in=R/8', 'forestgreen'),
        (0.5, 0.25, 'h_out=R/2, h_in=R/4', 'green'),
        (0.25, 0.25, 'h_out=R/4, h_in=R/4', 'darkgreen')
    ]
    
    for idx, (o_factor, i_factor, title, color) in enumerate(biconcave_configs):
        ax = fig.add_subplot(3, 4, idx+5, projection='3d')
        
        # Создаем сетку с высоким разрешением
        theta = np.linspace(0, 2*np.pi, 60)
        r = np.linspace(0, R_typical, 40)
        R_grid, Theta_grid = np.meshgrid(r, theta)
        X = R_grid * np.cos(Theta_grid)
        Y = R_grid * np.sin(Theta_grid)
        
        r_norm = R_grid / R_typical
        outer_height = R_typical * o_factor
        inner_height = R_typical * i_factor
        
        # Настоящая вогнутость: толщина уменьшается к центру
        # На краю: outer_height, в центре: inner_height
        thickness = outer_height - (outer_height - inner_height) * (1 - r_norm**2)**2
        Z_upper = thickness / 2
        Z_lower = -thickness / 2
        
        # Верхняя и нижняя поверхности с сеткой
        ax.plot_surface(X, Y, Z_upper, color=color, alpha=0.4, edgecolor='darkgreen', linewidth=0.5)
        ax.plot_surface(X, Y, Z_lower, color=color, alpha=0.4, edgecolor='darkgreen', linewidth=0.5)
        
        # Боковая граница по краю
        theta_edge = np.linspace(0, 2*np.pi, 80)
        r_edge = R_typical * np.ones_like(theta_edge)
        x_edge = r_edge * np.cos(theta_edge)
        y_edge = r_edge * np.sin(theta_edge)
        edge_thickness = outer_height  # на краю максимальная толщина
        z_upper_edge = edge_thickness / 2
        z_lower_edge = -edge_thickness / 2
        
        # Создаем боковую стенку
        for i in range(len(theta_edge)-1):
            ax.plot([x_edge[i], x_edge[i], x_edge[i+1], x_edge[i+1], x_edge[i]],
                   [y_edge[i], y_edge[i], y_edge[i+1], y_edge[i+1], y_edge[i]],
                   [z_lower_edge, z_upper_edge, z_upper_edge, z_lower_edge, z_lower_edge],
                   color='darkgreen', linewidth=1.2, alpha=0.8)
        
        ax.set_title(f'Двояковогнутый\n{title}', fontsize=9, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-R_typical-1, R_typical+1])
        ax.set_ylim([-R_typical-1, R_typical+1])
        ax.set_zlim([-R_typical-1, R_typical+1])
        ax.view_init(elev=25, azim=45)
    
    # 4. Эллипсоид - полные поверхности с сеткой
    angles_to_show = [0, angle, 90, 45]
    angles_to_show = list(set(angles_to_show))[:4]
    colors_ellipsoid = ['red', 'coral', 'tomato', 'salmon']
    
    for idx, (show_angle, color) in enumerate(zip(angles_to_show[:4], colors_ellipsoid)):
        ax = fig.add_subplot(3, 4, idx+9, projection='3d')
        
        u = np.linspace(0, 2*np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        u, v = np.meshgrid(u, v)
        
        a, b, c = R_typical, R_typical/2.5, R_typical/2.5
        
        x_orig = a * np.sin(v) * np.cos(u)
        y_orig = b * np.sin(v) * np.sin(u)
        z_orig = c * np.cos(v)
        
        angle_rad = np.radians(show_angle)
        x = x_orig * np.cos(angle_rad) - y_orig * np.sin(angle_rad)
        y = x_orig * np.sin(angle_rad) + y_orig * np.cos(angle_rad)
        z = z_orig
        
        ax.plot_surface(x, y, z, color=color, alpha=0.3, edgecolor='darkred', linewidth=0.5)
        
        highlight = " (заданный)" if show_angle == angle else ""
        ax.set_title(f'Эллипсоид, угол={show_angle}°{highlight}', fontsize=10, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-R_typical-1, R_typical+1])
        ax.set_ylim([-R_typical-1, R_typical+1])
        ax.set_zlim([-R_typical-1, R_typical+1])
        ax.view_init(elev=25, azim=45)
    
    plt.suptitle(f'3D модели частиц (заданный угол поворота эллипсоида: {angle}°)', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def plot_diffraction_patterns_with_extrema(theta_vals, sphere_result, disk_results, biconcave_results, 
                                          ellipsoid_results_long, ellipsoid_results_short, 
                                          user_angle_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # График 1: Длинная ось эллипсоида
    ax1 = axes[0, 0]
    
    # Сфера - розовый
    ax1.plot(theta_vals, sphere_result['intensity'], color='#FF69B4', 
            linewidth=2, label='Сфера', alpha=0.9)
    if sphere_result['min_theta'] is not None:
        ax1.plot(sphere_result['min_theta'], sphere_result['min_intensity'], 
                'v', color='#FF69B4', markersize=8, markeredgewidth=1.5)
    if sphere_result['max_theta'] is not None:
        ax1.plot(sphere_result['max_theta'], sphere_result['max_intensity'], 
                '^', color='#FF69B4', markersize=8, markeredgewidth=1.5)
    
    # Плоские диски - разные чистые цвета
    disk_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
    for (name, data), color in zip(disk_results.items(), disk_colors):
        ax1.plot(theta_vals, data['intensity'], color=color, 
                linewidth=2, label=f'Плоский диск {name}', alpha=0.8)
        if data['min_theta'] is not None:
            ax1.plot(data['min_theta'], data['min_intensity'], 'v', 
                    color=color, markersize=7)
        if data['max_theta'] is not None:
            ax1.plot(data['max_theta'], data['max_intensity'], '^', 
                    color=color, markersize=7)
    
    # Двояковогнутые диски - разные чистые цвета
    biconcave_colors = ['#FFA500', '#800080', '#00FFFF', '#FF00FF', '#A52A2A', '#808080']
    for (name, data), color in zip(biconcave_results.items(), biconcave_colors):
        ax1.plot(theta_vals, data['intensity'], color=color, 
                linewidth=2, label=f'Двояковогнутый {name}', alpha=0.8)
        if data['min_theta'] is not None:
            ax1.plot(data['min_theta'], data['min_intensity'], 'v', 
                    color=color, markersize=7)
        if data['max_theta'] is not None:
            ax1.plot(data['max_theta'], data['max_intensity'], '^', 
                    color=color, markersize=7)
    
    # Эллипсоид с длинной осью - черный
    ax1.plot(theta_vals, ellipsoid_results_long['intensity'], color='#000000', 
            linewidth=2, label='Эллипсоид (длинная ось)', alpha=0.9)
    if ellipsoid_results_long['min_theta'] is not None:
        ax1.plot(ellipsoid_results_long['min_theta'], ellipsoid_results_long['min_intensity'], 
                'v', color='#000000', markersize=10, markeredgewidth=2)
    if ellipsoid_results_long['max_theta'] is not None:
        ax1.plot(ellipsoid_results_long['max_theta'], ellipsoid_results_long['max_intensity'], 
                '^', color='#000000', markersize=10, markeredgewidth=2)
    
    ax1.set_xlabel('Угол θ (рад)', fontsize=12)
    ax1.set_ylabel('Интенсивность I(θ)', fontsize=12)
    ax1.set_title('Дифракция: все фигуры + эллипс (длинная ось)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=6, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 0.3])
    ax1.set_ylim([0, 1.1])
    
    # График 2: Короткая ось эллипсоида
    ax2 = axes[0, 1]
    
    # Сфера - розовый
    ax2.plot(theta_vals, sphere_result['intensity'], color='#FF69B4', 
            linewidth=2, label='Сфера', alpha=0.9)
    if sphere_result['min_theta'] is not None:
        ax2.plot(sphere_result['min_theta'], sphere_result['min_intensity'], 
                'v', color='#FF69B4', markersize=8, markeredgewidth=1.5)
    if sphere_result['max_theta'] is not None:
        ax2.plot(sphere_result['max_theta'], sphere_result['max_intensity'], 
                '^', color='#FF69B4', markersize=8, markeredgewidth=1.5)
    
    # Плоские диски
    for (name, data), color in zip(disk_results.items(), disk_colors):
        ax2.plot(theta_vals, data['intensity'], color=color, 
                linewidth=2, label=f'Плоский диск {name}', alpha=0.8)
        if data['min_theta'] is not None:
            ax2.plot(data['min_theta'], data['min_intensity'], 'v', 
                    color=color, markersize=7)
        if data['max_theta'] is not None:
            ax2.plot(data['max_theta'], data['max_intensity'], '^', 
                    color=color, markersize=7)
    
    # Двояковогнутые диски
    for (name, data), color in zip(biconcave_results.items(), biconcave_colors):
        ax2.plot(theta_vals, data['intensity'], color=color, 
                linewidth=2, label=f'Двояковогнутый {name}', alpha=0.8)
        if data['min_theta'] is not None:
            ax2.plot(data['min_theta'], data['min_intensity'], 'v', 
                    color=color, markersize=7)
        if data['max_theta'] is not None:
            ax2.plot(data['max_theta'], data['max_intensity'], '^', 
                    color=color, markersize=7)
    
    # Эллипсоид с короткой осью - черный
    ax2.plot(theta_vals, ellipsoid_results_short['intensity'], color='#000000', 
            linewidth=2, label='Эллипсоид (короткая ось)', alpha=0.9)
    if ellipsoid_results_short['min_theta'] is not None:
        ax2.plot(ellipsoid_results_short['min_theta'], ellipsoid_results_short['min_intensity'], 
                'v', color='#000000', markersize=10, markeredgewidth=2)
    if ellipsoid_results_short['max_theta'] is not None:
        ax2.plot(ellipsoid_results_short['max_theta'], ellipsoid_results_short['max_intensity'], 
                '^', color='#000000', markersize=10, markeredgewidth=2)
    
    ax2.set_xlabel('Угол θ (рад)', fontsize=12)
    ax2.set_ylabel('Интенсивность I(θ)', fontsize=12)
    ax2.set_title('Дифракция: все фигуры + эллипс (короткая ось)', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=6, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.3])
    ax2.set_ylim([0, 1.1])
    
    # График 3: Эллипсоид под заданным углом
    ax3 = axes[1, 0]
    
    # Сфера для сравнения - розовый пунктир
    ax3.plot(theta_vals, sphere_result['intensity'], color='#FF69B4', 
            linewidth=2, label='Сфера', alpha=0.7, linestyle='--')
    
    # Эллипсоид под заданным углом - черный
    ax3.plot(theta_vals, user_angle_results['intensity'], color='#000000', 
            linewidth=2, label=f'Эллипсоид (угол {user_angle_results["angle"]}°)', alpha=0.9)
    if user_angle_results['min_theta'] is not None:
        ax3.plot(user_angle_results['min_theta'], user_angle_results['min_intensity'], 
                'v', markersize=10, markeredgewidth=2, color='#000000')
    if user_angle_results['max_theta'] is not None:
        ax3.plot(user_angle_results['max_theta'], user_angle_results['max_intensity'], 
                '^', markersize=10, markeredgewidth=2, color='#000000')
    
    ax3.set_xlabel('Угол θ (рад)', fontsize=12)
    ax3.set_ylabel('Интенсивность I(θ)', fontsize=12)
    ax3.set_title(f'Дифракция для заданного угла поворота {user_angle_results["angle"]}°', 
                 fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 0.3])
    ax3.set_ylim([0, 1.1])
    
    # График 4: Информация
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    info = f"""ПАРАМЕТРЫ РАСЧЕТА:
длина_волны = {lambda_wave} мкм
R = {R_typical} мкм
n_среды = {n_medium}
n_частицы = {n_particle}
Размер диполя = {dipole_size:.3f} мкм

СФЕРА:
θ_min = {sphere_result["min_theta"]:.4f} рад
I_min = {sphere_result["min_intensity"]:.6f}
θ_max = {sphere_result["max_theta"]:.4f} рад
I_max = {sphere_result["max_intensity"]:.6f}

ЗАДАННЫЙ УГОЛ: {user_angle_results["angle"]}°

ДЛИННАЯ ОСЬ:
θ_min = {ellipsoid_results_long["min_theta"]:.4f} рад
I_min = {ellipsoid_results_long["min_intensity"]:.6f}

КОРОТКАЯ ОСЬ:
θ_min = {ellipsoid_results_short["min_theta"]:.4f} рад
I_min = {ellipsoid_results_short["min_intensity"]:.6f}

ЗАДАННЫЙ УГОЛ:
θ_min = {user_angle_results["min_theta"]:.4f} рад
I_min = {user_angle_results["min_intensity"]:.6f}"""
    
    ax4.text(0.05, 0.95, info, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.suptitle('Дифракционные картины с отмеченными экстремумами', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def main():
    start_time = time.time()
    
    # Интерактивный ввод угла
    while True:
        try:
            user_angle = float(input("Введите угол поворота эллипсоида (от 0 до 90 градусов): "))
            if 0 <= user_angle <= 90:
                break
            else:
                print("Ошибка: угол должен быть в диапазоне от 0 до 90 градусов.")
        except ValueError:
            print("Ошибка: введите число.")
    
    theta_max = np.radians(25)
    theta_vals = np.linspace(0.001, theta_max, 500)
    
    
    print("\nСоздание сферы:")
    dipoles_sphere = create_sphere(R_typical)
    print(f"   Создано {len(dipoles_sphere)} диполей")
    
    I_sphere = adda_dda(theta_vals, dipoles_sphere)
    min_t_sphere, min_i_sphere, max_t_sphere, max_i_sphere = find_extrema(theta_vals, I_sphere)
    
    sphere_result = {
        'intensity': I_sphere,
        'min_theta': min_t_sphere,
        'min_intensity': min_i_sphere,
        'max_theta': max_t_sphere,
        'max_intensity': max_i_sphere,
        'dipoles': len(dipoles_sphere)
    }
    
    print(f"   Первый минимум: θ = {min_t_sphere:.4f} рад, I = {min_i_sphere:.6f}")
    print(f"   Первый максимум: θ = {max_t_sphere:.4f} рад, I = {max_i_sphere:.6f}")
    
    
    height_factors = [1, 0.5, 0.25, 0.125]
    disk_results = {}
    
    for h_factor in height_factors:
        print(f"\nСоздание плоского диска с h = R/{1/h_factor if h_factor<1 else 1}:")
        dipoles = create_flat_disk(R_typical, h_factor)
        print(f"   Создано {len(dipoles)} диполей")
        
        I = adda_dda(theta_vals, dipoles)
        min_t, min_i, max_t, max_i = find_extrema(theta_vals, I)
        
        disk_results[f'h={h_factor}R'] = {
            'intensity': I,
            'min_theta': min_t,
            'min_intensity': min_i,
            'max_theta': max_t,
            'max_intensity': max_i,
            'dipoles': len(dipoles)
        }
        
        print(f"   Первый минимум: θ = {min_t:.4f} рад, I = {min_i:.6f}")
        print(f"   Первый максимум: θ = {max_t:.4f} рад, I = {max_i:.6f}")
    
    
    outer_factors = [1, 0.5, 0.25]
    inner_factors = [0.25, 0.125]
    biconcave_results = {}
    
    for o_factor in outer_factors:
        for i_factor in inner_factors:
            print(f"\nСоздание двояковогнутого диска: внеш. h={o_factor}R, внутр. h={i_factor}R:")
            dipoles = create_biconcave_disk(R_typical, o_factor, i_factor)
            print(f"   Создано {len(dipoles)} диполей")
            
            I = adda_dda(theta_vals, dipoles)
            min_t, min_i, max_t, max_i = find_extrema(theta_vals, I)
            
            biconcave_results[f'out={o_factor}R_in={i_factor}R'] = {
                'intensity': I,
                'min_theta': min_t,
                'min_intensity': min_i,
                'max_theta': max_t,
                'max_intensity': max_i,
                'dipoles': len(dipoles)
            }
            
            print(f"   Первый минимум: θ = {min_t:.4f} рад, I = {min_i:.6f}")
            print(f"   Первый максимум: θ = {max_t:.4f} рад, I = {max_i:.6f}")
    
    # Расчет для заданного угла
    print(f"\nСоздание эллипсоида с углом поворота {user_angle}°:")
    dipoles_user = create_rotated_ellipsoid(R_typical, user_angle)
    print(f"   Создано {len(dipoles_user)} диполей")
    
    I_user = adda_dda(theta_vals, dipoles_user)
    min_t_user, min_i_user, max_t_user, max_i_user = find_extrema(theta_vals, I_user)
    
    user_angle_results = {
        'angle': user_angle,
        'intensity': I_user,
        'min_theta': min_t_user,
        'min_intensity': min_i_user,
        'max_theta': max_t_user,
        'max_intensity': max_i_user,
        'dipoles': len(dipoles_user)
    }
    
    print(f"   Первый минимум: θ = {min_t_user:.4f} рад, I = {min_i_user:.6f}")
    print(f"   Первый максимум: θ = {max_t_user:.4f} рад, I = {max_i_user:.6f}")
    
    
    # Для длинной оси (угол 0°, плоскость XZ)
    dipoles_long = create_rotated_ellipsoid(R_typical, 0)
    I_long = adda_dda(theta_vals, dipoles_long, plane='xz')
    min_t_long, min_i_long, max_t_long, max_i_long = find_extrema(theta_vals, I_long)
    ellipsoid_results_long = {
        'intensity': I_long,
        'min_theta': min_t_long,
        'min_intensity': min_i_long,
        'max_theta': max_t_long,
        'max_intensity': max_i_long,
        'dipoles': len(dipoles_long)
    }
    print(f"\nЭллипсоид (длинная ось):")
    print(f"   Первый минимум: θ = {min_t_long:.4f} рад, I = {min_i_long:.6f}")
    print(f"   Первый максимум: θ = {max_t_long:.4f} рад, I = {max_i_long:.6f}")
    
    dipoles_short = create_rotated_ellipsoid(R_typical, 90)
    I_short = adda_dda(theta_vals, dipoles_short, plane='yz')
    min_t_short, min_i_short, max_t_short, max_i_short = find_extrema(theta_vals, I_short)
    ellipsoid_results_short = {
        'intensity': I_short,
        'min_theta': min_t_short,
        'min_intensity': min_i_short,
        'max_theta': max_t_short,
        'max_intensity': max_i_short,
        'dipoles': len(dipoles_short)
    }
    print(f"\nЭллипсоид (короткая ось):")
    print(f"   Первый минимум: θ = {min_t_short:.4f} рад, I = {min_i_short:.6f}")
    print(f"   Первый максимум: θ = {max_t_short:.4f} рад, I = {max_i_short:.6f}")
    
    print("Построение 3D моделей...")
    plot_3d_models_with_angle(user_angle)
    
    print("\nПостроение дифракционных картин с отмеченными экстремумами...")
    plot_diffraction_patterns_with_extrema(theta_vals, sphere_result, disk_results, biconcave_results, 
                                          ellipsoid_results_long, ellipsoid_results_short,
                                          user_angle_results)
    
    total_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {total_time:.2f} секунд")
    
    print(f"{'Тип частицы':<35} {'θ_min (рад)':<12} {'I_min':<12} {'θ_max (рад)':<12} {'I_max':<12}")
    print("-"*85)
    
    print(f"{'Сфера':<35} {sphere_result['min_theta']:.4f}       {sphere_result['min_intensity']:.6f}   {sphere_result['max_theta']:.4f}       {sphere_result['max_intensity']:.6f}")
    
    for name, data in disk_results.items():
        print(f"{'Плоский диск ' + name:<35} {data['min_theta']:.4f}       {data['min_intensity']:.6f}   {data['max_theta']:.4f}       {data['max_intensity']:.6f}")
    
    for name, data in biconcave_results.items():
        print(f"{'Двояковогнутый ' + name:<35} {data['min_theta']:.4f}       {data['min_intensity']:.6f}   {data['max_theta']:.4f}       {data['max_intensity']:.6f}")
    
    print(f"{'Эллипсоид (длинная ось)':<35} {ellipsoid_results_long['min_theta']:.4f}       {ellipsoid_results_long['min_intensity']:.6f}   {ellipsoid_results_long['max_theta']:.4f}       {ellipsoid_results_long['max_intensity']:.6f}")
    print(f"{'Эллипсоид (короткая ось)':<35} {ellipsoid_results_short['min_theta']:.4f}       {ellipsoid_results_short['min_intensity']:.6f}   {ellipsoid_results_short['max_theta']:.4f}       {ellipsoid_results_short['max_intensity']:.6f}")
    print(f"{f'Эллипсоид (угол={user_angle}°)':<35} {user_angle_results['min_theta']:.4f}       {user_angle_results['min_intensity']:.6f}   {user_angle_results['max_theta']:.4f}       {user_angle_results['max_intensity']:.6f}")

if __name__ == "__main__":
    main()