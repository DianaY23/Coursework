# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import time


lambda_wave = 0.6328      
n_medium = 1.33           
n_particle = 1.05         
m = n_particle / n_medium
R_typical = 3.75          

# Параметры лазерного пучка
w0_laser = 200.0          
z_focus = 0.0             

points_per_lambda = 8
dipole_size = lambda_wave / points_per_lambda
print(f"Размер диполя: {dipole_size:.3f} мкм")
print(f"Точек на длину волны: {points_per_lambda}")


def gaussian_beam(dipoles, w0, z_focus=0, E0=1.0): #гауссовый пучок
    wavelength_med = lambda_wave / n_medium
    k_med = 2 * np.pi / wavelength_med
    zR = (np.pi * w0**2 * n_medium) / lambda_wave
    if zR < 1e-10:
        zR = 1e10
    
    x = dipoles[:, 0]
    y = dipoles[:, 1]
    z = dipoles[:, 2] - z_focus
    r2 = x**2 + y**2
    
    wz = w0 * np.sqrt(1 + (z / zR)**2)
    
    Rz = np.ones_like(z) * 1e10
    mask = np.abs(z) > 1e-6
    Rz[mask] = z[mask] * (1 + (zR / z[mask])**2)
    
    amplitude = E0 * (w0 / wz) * np.exp(-r2 / (wz**2))
    phase = -k_med * z - k_med * r2 / (2 * Rz) + np.arctan2(z, zR)
    
    return amplitude * np.exp(1j * phase)

def adda_dda_laser(theta_vals, dipoles, plane='xz'):
    if len(dipoles) == 0:
        return np.zeros_like(theta_vals)
    
    wavelength_med = lambda_wave / n_medium
    k_med = 2 * np.pi / wavelength_med
    alpha = dipole_size**3 * (m**2 - 1) / (m**2 + 2)
    
    E_inc = gaussian_beam(dipoles, w0_laser, z_focus) #источник
    N = len(dipoles)
    I_arr = []
    
    k_inc = np.array([0, 0, k_med])
    
    for theta in theta_vals:
        if theta == 0:
            I_arr.append(1.0)
            continue
        
        if plane == 'xz':
            k_sca = np.array([k_med * np.sin(theta), 0, k_med * np.cos(theta)])
        else:
            k_sca = np.array([0, k_med * np.sin(theta), k_med * np.cos(theta)])
        
        q = k_sca - k_inc
        
        phases = np.exp(1j * np.dot(dipoles, q))
        structure_factor = np.abs(np.sum(phases))**2 / N**2
        
        intensity = structure_factor
        I_arr.append(intensity)
    
    I_arr = np.array(I_arr)
    if np.max(I_arr) > 0:
        I_arr = I_arr / np.max(I_arr)
    return I_arr

#Функция Эйри (теория)
def airy_diffraction(theta_vals, radius):
    wavelength_med = lambda_wave / n_medium
    k_med = 2 * np.pi / wavelength_med
    x = k_med * radius * np.sin(theta_vals)
    x = np.where(x == 0, 1e-10, x)
    intensity = (2 * j1(x) / x)**2
    return intensity / intensity[0] if intensity[0] > 0 else intensity


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
                
                r = np.sqrt(x**2 + y**2)
                
                if r <= radius and abs(z) <= height/2:
                    dipoles.append([x, y, z])
    
    return np.array(dipoles)

def create_biconcave_disk(radius, outer_height_factor, inner_height_factor):
    dipoles = []
    outer_height = radius * outer_height_factor
    inner_height = radius * inner_height_factor
    
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
#кривая скалака
def create_skalak_erythrocyte(radius, max_height=1.4):
    dipoles = []
    c0, c2, c4 = 0.0518, 2.0026, -1.0544
    
    size = int(2 * radius / dipole_size) + 1
    
    for i in range(-size, size+1):
        for j in range(-size, size+1):
            x = i * dipole_size
            y = j * dipole_size
            r = np.sqrt(x**2 + y**2)
            
            if r <= radius:
                r_norm = r / radius
                z_surface = max_height * np.sqrt(1 - r_norm**2) * (c0 + c2*r_norm**2 + c4*r_norm**4)
                z_surface = max(z_surface, 0.15)
                
                num_layers = max(1, int(2 * z_surface / dipole_size))
                
                for k in range(-num_layers, num_layers + 1):
                    z = k * dipole_size
                    if abs(z) <= z_surface:
                        dipoles.append([x, y, z])
    
    return np.array(dipoles)
#ансабль эритроцитов
def create_ensemble_in_plane(centers_xy, z_layer=0):
    all_dipoles = []
    
    for cx, cy in centers_xy:
        base_dipoles = create_skalak_erythrocyte(R_typical)
        for d in base_dipoles:
            all_dipoles.append([d[0] + cx, d[1] + cy, d[2] + z_layer])
    
    return np.array(all_dipoles)

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
        if theta_vals[idx] > 0.01:
            first_max = theta_vals[idx]
            first_max_val = intensity[idx]
            break
    
    return first_min, first_min_val, first_max, first_max_val


def plot_3d_objects():
    fig = plt.figure(figsize=(24, 12))
    
    # сфера
    ax1 = fig.add_subplot(2, 5, 1, projection='3d')
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    u, v = np.meshgrid(u, v)
    x = R_typical * np.sin(v) * np.cos(u)
    y = R_typical * np.sin(v) * np.sin(u)
    z = R_typical * np.cos(v)
    ax1.plot_surface(x, y, z, color='cyan', alpha=0.5, edgecolor='blue', linewidth=0.5)
    ax1.set_title('Сфера', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X, мкм')
    ax1.set_ylabel('Y, мкм')
    ax1.set_zlabel('Z, мкм')
    ax1.view_init(elev=25, azim=45)
    
    # плоский диск с рызными высотами 
    ax2 = fig.add_subplot(2, 5, 2, projection='3d')
    height = R_typical * 1.0
    theta_side = np.linspace(0, 2*np.pi, 50)
    z_side = np.linspace(-height/2, height/2, 20)
    Theta_side, Z_side = np.meshgrid(theta_side, z_side)
    X_side = R_typical * np.cos(Theta_side)
    Y_side = R_typical * np.sin(Theta_side)
    theta_top = np.linspace(0, 2*np.pi, 50)
    r_top = np.linspace(0, R_typical, 30)
    R_top, Theta_top = np.meshgrid(r_top, theta_top)
    X_top = R_top * np.cos(Theta_top)
    Y_top = R_top * np.sin(Theta_top)
    Z_top = np.ones_like(X_top) * height/2
    Z_bottom = np.ones_like(X_top) * -height/2
    ax2.plot_surface(X_side, Y_side, Z_side, color='blue', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax2.plot_surface(X_top, Y_top, Z_top, color='blue', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax2.plot_surface(X_top, Y_top, Z_bottom, color='blue', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax2.set_title('Плоский диск (h = R)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X, мкм')
    ax2.set_ylabel('Y, мкм')
    ax2.set_zlabel('Z, мкм')
    ax2.view_init(elev=25, azim=45)
    

    ax3 = fig.add_subplot(2, 5, 3, projection='3d')
    height = R_typical * 0.5
    theta_side = np.linspace(0, 2*np.pi, 50)
    z_side = np.linspace(-height/2, height/2, 20)
    Theta_side, Z_side = np.meshgrid(theta_side, z_side)
    X_side = R_typical * np.cos(Theta_side)
    Y_side = R_typical * np.sin(Theta_side)
    theta_top = np.linspace(0, 2*np.pi, 50)
    r_top = np.linspace(0, R_typical, 30)
    R_top, Theta_top = np.meshgrid(r_top, theta_top)
    X_top = R_top * np.cos(Theta_top)
    Y_top = R_top * np.sin(Theta_top)
    Z_top = np.ones_like(X_top) * height/2
    Z_bottom = np.ones_like(X_top) * -height/2
    ax3.plot_surface(X_side, Y_side, Z_side, color='royalblue', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax3.plot_surface(X_top, Y_top, Z_top, color='royalblue', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax3.plot_surface(X_top, Y_top, Z_bottom, color='royalblue', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax3.set_title('Плоский диск (h = R/2)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X, мкм')
    ax3.set_ylabel('Y, мкм')
    ax3.set_zlabel('Z, мкм')
    ax3.view_init(elev=25, azim=45)
    

    ax4 = fig.add_subplot(2, 5, 4, projection='3d')
    height = R_typical * 0.25
    theta_side = np.linspace(0, 2*np.pi, 50)
    z_side = np.linspace(-height/2, height/2, 20)
    Theta_side, Z_side = np.meshgrid(theta_side, z_side)
    X_side = R_typical * np.cos(Theta_side)
    Y_side = R_typical * np.sin(Theta_side)
    theta_top = np.linspace(0, 2*np.pi, 50)
    r_top = np.linspace(0, R_typical, 30)
    R_top, Theta_top = np.meshgrid(r_top, theta_top)
    X_top = R_top * np.cos(Theta_top)
    Y_top = R_top * np.sin(Theta_top)
    Z_top = np.ones_like(X_top) * height/2
    Z_bottom = np.ones_like(X_top) * -height/2
    ax4.plot_surface(X_side, Y_side, Z_side, color='cornflowerblue', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax4.plot_surface(X_top, Y_top, Z_top, color='cornflowerblue', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax4.plot_surface(X_top, Y_top, Z_bottom, color='cornflowerblue', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax4.set_title('Плоский диск (h = R/4)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X, мкм')
    ax4.set_ylabel('Y, мкм')
    ax4.set_zlabel('Z, мкм')
    ax4.view_init(elev=25, azim=45)
    

    ax5 = fig.add_subplot(2, 5, 5, projection='3d')
    height = R_typical * 0.125
    theta_side = np.linspace(0, 2*np.pi, 50)
    z_side = np.linspace(-height/2, height/2, 20)
    Theta_side, Z_side = np.meshgrid(theta_side, z_side)
    X_side = R_typical * np.cos(Theta_side)
    Y_side = R_typical * np.sin(Theta_side)
    theta_top = np.linspace(0, 2*np.pi, 50)
    r_top = np.linspace(0, R_typical, 30)
    R_top, Theta_top = np.meshgrid(r_top, theta_top)
    X_top = R_top * np.cos(Theta_top)
    Y_top = R_top * np.sin(Theta_top)
    Z_top = np.ones_like(X_top) * height/2
    Z_bottom = np.ones_like(X_top) * -height/2
    ax5.plot_surface(X_side, Y_side, Z_side, color='navy', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax5.plot_surface(X_top, Y_top, Z_top, color='navy', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax5.plot_surface(X_top, Y_top, Z_bottom, color='navy', alpha=0.4, edgecolor='darkblue', linewidth=0.5)
    ax5.set_title('Плоский диск (h = R/8)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('X, мкм')
    ax5.set_ylabel('Y, мкм')
    ax5.set_zlabel('Z, мкм')
    ax5.view_init(elev=25, azim=45)
    
    # двояковогнутый диск
    ax6 = fig.add_subplot(2, 5, 6, projection='3d')
    theta = np.linspace(0, 2*np.pi, 60)
    r = np.linspace(0, R_typical, 40)
    R_grid, Theta_grid = np.meshgrid(r, theta)
    X = R_grid * np.cos(Theta_grid)
    Y = R_grid * np.sin(Theta_grid)
    outer_height = R_typical * 1.0
    inner_height = R_typical * 0.25
    r_norm = R_grid / R_typical
    thickness = outer_height - (outer_height - inner_height) * (1 - r_norm**2)**2
    Z_upper = thickness / 2
    Z_lower = -thickness / 2
    ax6.plot_surface(X, Y, Z_upper, color='green', alpha=0.5, edgecolor='darkgreen', linewidth=0.5)
    ax6.plot_surface(X, Y, Z_lower, color='green', alpha=0.5, edgecolor='darkgreen', linewidth=0.5)
    
    theta_edge = np.linspace(0, 2*np.pi, 80)
    x_edge = R_typical * np.cos(theta_edge)
    y_edge = R_typical * np.sin(theta_edge)
    edge_thickness = outer_height
    z_upper_edge = edge_thickness / 2
    z_lower_edge = -edge_thickness / 2
    
    for i in range(len(theta_edge)-1):
        ax6.plot([x_edge[i], x_edge[i], x_edge[i+1], x_edge[i+1], x_edge[i]],
                [y_edge[i], y_edge[i], y_edge[i+1], y_edge[i+1], y_edge[i]],
                [z_lower_edge, z_upper_edge, z_upper_edge, z_lower_edge, z_lower_edge],
                color='darkgreen', linewidth=1.2, alpha=0.8)
    
    ax6.set_title('Двояковогнутый\n(out=R, in=R/4)', fontsize=11, fontweight='bold')
    ax6.set_xlabel('X, мкм')
    ax6.set_ylabel('Y, мкм')
    ax6.set_zlabel('Z, мкм')
    ax6.view_init(elev=25, azim=45)
    
    ax7 = fig.add_subplot(2, 5, 7, projection='3d')
    thickness = outer_height - (outer_height - R_typical*0.125) * (1 - r_norm**2)**2
    Z_upper = thickness / 2
    Z_lower = -thickness / 2
    ax7.plot_surface(X, Y, Z_upper, color='limegreen', alpha=0.5, edgecolor='darkgreen', linewidth=0.5)
    ax7.plot_surface(X, Y, Z_lower, color='limegreen', alpha=0.5, edgecolor='darkgreen', linewidth=0.5)
    
    for i in range(len(theta_edge)-1):
        ax7.plot([x_edge[i], x_edge[i], x_edge[i+1], x_edge[i+1], x_edge[i]],
                [y_edge[i], y_edge[i], y_edge[i+1], y_edge[i+1], y_edge[i]],
                [z_lower_edge, z_upper_edge, z_upper_edge, z_lower_edge, z_lower_edge],
                color='darkgreen', linewidth=1.2, alpha=0.8)
    
    ax7.set_title('Двояковогнутый\n(out=R, in=R/8)', fontsize=11, fontweight='bold')
    ax7.set_xlabel('X, мкм')
    ax7.set_ylabel('Y, мкм')
    ax7.set_zlabel('Z, мкм')
    ax7.view_init(elev=25, azim=45)
    
    ax8 = fig.add_subplot(2, 5, 8, projection='3d')
    outer_height2 = R_typical * 0.5
    inner_height2 = R_typical * 0.25
    thickness2 = outer_height2 - (outer_height2 - inner_height2) * (1 - r_norm**2)**2
    Z_upper2 = thickness2 / 2
    Z_lower2 = -thickness2 / 2
    ax8.plot_surface(X, Y, Z_upper2, color='forestgreen', alpha=0.5, edgecolor='darkgreen', linewidth=0.5)
    ax8.plot_surface(X, Y, Z_lower2, color='forestgreen', alpha=0.5, edgecolor='darkgreen', linewidth=0.5)
    
    edge_thickness2 = outer_height2
    z_upper_edge2 = edge_thickness2 / 2
    z_lower_edge2 = -edge_thickness2 / 2
    
    for i in range(len(theta_edge)-1):
        ax8.plot([x_edge[i], x_edge[i], x_edge[i+1], x_edge[i+1], x_edge[i]],
                [y_edge[i], y_edge[i], y_edge[i+1], y_edge[i+1], y_edge[i]],
                [z_lower_edge2, z_upper_edge2, z_upper_edge2, z_lower_edge2, z_lower_edge2],
                color='darkgreen', linewidth=1.2, alpha=0.8)
    
    ax8.set_title('Двояковогнутый\n(out=R/2, in=R/4)', fontsize=11, fontweight='bold')
    ax8.set_xlabel('X, мкм')
    ax8.set_ylabel('Y, мкм')
    ax8.set_zlabel('Z, мкм')
    ax8.view_init(elev=25, azim=45)
    
    # эллипс длинная ось
    ax9 = fig.add_subplot(2, 5, 9, projection='3d')
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    u, v = np.meshgrid(u, v)
    a, b, c = R_typical, R_typical/2.5, R_typical/2.5
    x_ell = a * np.sin(v) * np.cos(u)
    y_ell = b * np.sin(v) * np.sin(u)
    z_ell = c * np.cos(v)
    ax9.plot_surface(x_ell, y_ell, z_ell, color='red', alpha=0.5, edgecolor='darkred', linewidth=0.5)
    ax9.set_title('Эллипсоид\n(длинная ось, 0°)', fontsize=11, fontweight='bold')
    ax9.set_xlabel('X, мкм')
    ax9.set_ylabel('Y, мкм')
    ax9.set_zlabel('Z, мкм')
    ax9.view_init(elev=25, azim=45)
    
    # эритроцит
    ax10 = fig.add_subplot(2, 5, 10, projection='3d')
    theta_surf = np.linspace(0, 2*np.pi, 60)
    r_surf = np.linspace(0, R_typical, 50)
    R_surf, Theta_surf = np.meshgrid(r_surf, theta_surf)
    X_surf = R_surf * np.cos(Theta_surf)
    Y_surf = R_surf * np.sin(Theta_surf)
    c0, c2, c4 = 0.0518, 2.0026, -1.0544
    r_norm_surf = R_surf / R_typical
    z_profile = 1.4 * np.sqrt(1 - r_norm_surf**2) * (c0 + c2*r_norm_surf**2 + c4*r_norm_surf**4)
    z_profile = np.maximum(z_profile, 0.15)
    ax10.plot_surface(X_surf, Y_surf, z_profile, color='purple', alpha=0.6, edgecolor='darkviolet', linewidth=0.3)
    ax10.plot_surface(X_surf, Y_surf, -z_profile, color='purple', alpha=0.6, edgecolor='darkviolet', linewidth=0.3)
    ax10.set_title('Эритроцит', fontsize=11, fontweight='bold')
    ax10.set_xlabel('X, мкм')
    ax10.set_ylabel('Y, мкм')
    ax10.set_zlabel('Z, мкм')
    ax10.view_init(elev=25, azim=45)
    
    plt.suptitle('3D модели частиц (разных форм)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_diffraction_patterns(theta_vals, sphere_result, disk_results, biconcave_results, 
                              ellipsoid_long, ellipsoid_short, erythrocyte_result, ensemble_result, airy_result):
    
    # длинная ось эллипса
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.plot(theta_vals*180/np.pi, sphere_result['intensity'], 'm-', lw=2.5, label='Сфера (DDA)')
    
    disk_colors = ['r', 'g', 'b', 'orange']
    disk_labels = ['h=R', 'h=R/2', 'h=R/4', 'h=R/8']
    for (name, data), color, label in zip(disk_results.items(), disk_colors, disk_labels):
        ax1.plot(theta_vals*180/np.pi, data['intensity'], color=color, lw=1.8, label=f'Плоский диск {label} (DDA)')
    
    biconcave_colors = ['brown', 'olive', 'teal', 'gold']
    biconcave_labels = ['out=R,in=R/4', 'out=R,in=R/8', 'out=R/2,in=R/4', 'out=R/4,in=R/4']
    for (name, data), color, label in zip(biconcave_results.items(), biconcave_colors, biconcave_labels):
        ax1.plot(theta_vals*180/np.pi, data['intensity'], color=color, lw=1.8, label=f'Двояковогнутый {label} (DDA)')
    
    ax1.plot(theta_vals*180/np.pi, ellipsoid_long['intensity'], 'k-', lw=2.5, label='Эллипсоид 0° (DDA)')
    ax1.plot(theta_vals*180/np.pi, erythrocyte_result['intensity'], 'darkviolet', lw=2.5, label='Эритроцит (DDA)')
    ax1.plot(theta_vals*180/np.pi, ensemble_result['intensity'], 'orange', lw=2.5, label='3 эритроцита в XY (DDA)')
    ax1.plot(theta_vals*180/np.pi, airy_result, 'c--', lw=2, label='Теория Эйри')
    
    ax1.set_xlabel('Угол (градусы)', fontsize=12)
    ax1.set_ylabel('Интенсивность I', fontsize=12)
    ax1.set_title('Дифракционная картина: с длинной осью эллипсоида', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 12])
    ax1.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.show()
    
    # ГРАФИК 2: Короткая ось эллипсоида (90°) + ВСЕ ОБЪЕКТЫ
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    ax2.plot(theta_vals*180/np.pi, sphere_result['intensity'], 'm-', lw=2.5, label='Сфера (DDA)')
    
    for (name, data), color, label in zip(disk_results.items(), disk_colors, disk_labels):
        ax2.plot(theta_vals*180/np.pi, data['intensity'], color=color, lw=1.8, label=f'Плоский диск {label} (DDA)')
    
    for (name, data), color, label in zip(biconcave_results.items(), biconcave_colors, biconcave_labels):
        ax2.plot(theta_vals*180/np.pi, data['intensity'], color=color, lw=1.8, label=f'Двояковогнутый {label} (DDA)')
    
    ax2.plot(theta_vals*180/np.pi, ellipsoid_short['intensity'], 'k-', lw=2.5, label='Эллипсоид 90° (DDA)')
    ax2.plot(theta_vals*180/np.pi, erythrocyte_result['intensity'], 'darkviolet', lw=2.5, label='Эритроцит (DDA)')
    ax2.plot(theta_vals*180/np.pi, ensemble_result['intensity'], 'orange', lw=2.5, label='3 эритроцита в XY (DDA)')
    ax2.plot(theta_vals*180/np.pi, airy_result, 'c--', lw=2, label='Теория Эйри')
    
    ax2.set_xlabel('Угол (градусы)', fontsize=12)
    ax2.set_ylabel('Интенсивность I', fontsize=12)
    ax2.set_title('Дифракционная картина: с короткой осью эллипсоида', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 12])
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.show()


def plot_comparison_airy(theta_vals, sphere_result, ellipsoid_long, ellipsoid_short, airy_result):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Сравнение для сферы
    ax1 = axes[0]
    ax1.plot(theta_vals*180/np.pi, sphere_result['intensity'], 'b-', lw=2.5, label='DDA (сфера)')
    ax1.plot(theta_vals*180/np.pi, airy_result, 'r--', lw=2.5, label='Теория Эйри')
    ax1.set_xlabel('Угол (градусы)', fontsize=12)
    ax1.set_ylabel('Интенсивность I', fontsize=12)
    ax1.set_title('Сравнение DDA и теории Эйри для сферы', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 10])
    ax1.set_ylim([0, 1.05])
    
    # Сравнение для эллипсоидов
    ax2 = axes[1]
    ax2.plot(theta_vals*180/np.pi, ellipsoid_long['intensity'], 'g-', lw=2, label='DDA (эллипсоид 0°)')
    ax2.plot(theta_vals*180/np.pi, ellipsoid_short['intensity'], 'b-', lw=2, label='DDA (эллипсоид 90°)')
    ax2.plot(theta_vals*180/np.pi, airy_result, 'r--', lw=2.5, label='Теория Эйри')
    ax2.set_xlabel('Угол (градусы)', fontsize=12)
    ax2.set_ylabel('Интенсивность I', fontsize=12)
    ax2.set_title('Сравнение DDA для эллипсоидов и теории Эйри', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 10])
    ax2.set_ylim([0, 1.05])
    
    plt.suptitle('Сравнение метода DDA с теорией Эйри', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()



def plot_erythrocytes_in_xy_plane():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    centers = [(-15, -15), (0, 0), (15, 15)]
    colors = ['red', 'green', 'blue']
    labels = ['Эритроцит 1\n(-15, -15)', 'Эритроцит 2\n(0, 0)', 'Эритроцит 3\n(15, 15)']
    
    R = R_typical
    
    for (cx, cy), color, label in zip(centers, colors, labels):
        circle = plt.Circle((cx, cy), R, color=color, alpha=0.5, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.plot(cx, cy, 'o', color=color, markersize=8, markeredgecolor='black')
        ax.annotate(label, (cx, cy), xytext=(5, 5), textcoords='offset points', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_xlabel('X, мкм', fontsize=12)
    ax.set_ylabel('Y, мкм', fontsize=12)
    ax.set_title('3 эритроцита в плоскости XY', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    legend_elements = [Patch(facecolor='red', alpha=0.5, label='Эритроцит 1 (-15,-15)'),
                       Patch(facecolor='green', alpha=0.5, label='Эритроцит 2 (0,0)'),
                       Patch(facecolor='blue', alpha=0.5, label='Эритроцит 3 (15,15)')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()



def main():
    start_time = time.time()
    
    # Ввод угла
    while True:
        try:
            user_angle = float(input("Введите угол поворота эллипсоида (0-90): "))
            if 0 <= user_angle <= 90:
                break
            print("Ошибка: угол от 0 до 90")
        except ValueError:
            print("Ошибка: введите число")
    
    theta_max = np.radians(25)
    theta_vals = np.linspace(0.001, theta_max, 500)
    
    # Сфера
    print("Сфера")
    dip_sphere = create_sphere(R_typical)
    print(f"Диполей: {len(dip_sphere)}")
    I_sphere = adda_dda_laser(theta_vals, dip_sphere)
    min_t, min_i, max_t, max_i = find_extrema(theta_vals, I_sphere)
    sphere_result = {
        'intensity': I_sphere, 
        'min_theta': min_t, 'min_intensity': min_i,
        'max_theta': max_t, 'max_intensity': max_i
    }
    print(f"θ_min = {min_t:.4f} рад ({min_t*180/np.pi:.2f}°)")
    print(f"θ_max = {max_t:.4f} рад ({max_t*180/np.pi:.2f}°)")
    
    # Плоские диски
    print("Плоский диск")
    disk_results = {}
    for h in [1, 0.5, 0.25, 0.125]:
        dip = create_flat_disk(R_typical, h)
        I = adda_dda_laser(theta_vals, dip)
        mn, mi, mx, ma = find_extrema(theta_vals, I)
        disk_results[f'h={h}R'] = {
            'intensity': I, 
            'min_theta': mn, 'min_intensity': mi,
            'max_theta': mx, 'max_intensity': ma
        }
        print(f"h={h}R: диполей={len(dip)}")
        print(f"   угол_min = {mn:.4f} рад ({mn*180/np.pi:.2f}°)")
        print(f"   угол_max = {mx:.4f} рад ({mx*180/np.pi:.2f}°)")
    
    # Двояковогнутые
    print("Двояковогнутый диск")
    biconcave_results = {}
    for out in [1, 0.5]:
        for inn in [0.25, 0.125]:
            dip = create_biconcave_disk(R_typical, out, inn)
            I = adda_dda_laser(theta_vals, dip)
            mn, mi, mx, ma = find_extrema(theta_vals, I)
            name = f'out={out}R,in={inn}R'
            biconcave_results[name] = {
                'intensity': I,
                'min_theta': mn, 'min_intensity': mi,
                'max_theta': mx, 'max_intensity': ma
            }
            print(f"{name}: диполей={len(dip)}")
            print(f"   угол_min = {mn:.4f} рад ({mn*180/np.pi:.2f}°)")
            print(f"   угол_max = {mx:.4f} рад ({mx*180/np.pi:.2f}°)")
    
    # Эллипсоид (0°)
    dip_long = create_rotated_ellipsoid(R_typical, 0)
    I_long = adda_dda_laser(theta_vals, dip_long, plane='xz')
    mn_l, mi_l, mx_l, ma_l = find_extrema(theta_vals, I_long)
    long_result = {
        'intensity': I_long,
        'min_theta': mn_l, 'min_intensity': mi_l,
        'max_theta': mx_l, 'max_intensity': ma_l
    }
    print(f"Эллипсоид 0°: угол_min = {mn_l:.4f} рад ({mn_l*180/np.pi:.2f}°)")
    print(f"угол_max = {mx_l:.4f} рад ({mx_l*180/np.pi:.2f}°)")
    
    # Эллипсоид (90°)
    print("Эллипсоид 90°")
    dip_short = create_rotated_ellipsoid(R_typical, 90)
    I_short = adda_dda_laser(theta_vals, dip_short, plane='yz')
    mn_s, mi_s, mx_s, ma_s = find_extrema(theta_vals, I_short)
    short_result = {
        'intensity': I_short,
        'min_theta': mn_s, 'min_intensity': mi_s,
        'max_theta': mx_s, 'max_intensity': ma_s
    }
    print(f"угол_min = {mn_s:.4f} рад ({mn_s*180/np.pi:.2f}°)")
    print(f"угол_max = {mx_s:.4f} рад ({mx_s*180/np.pi:.2f}°)")
    
    # Эритроцит (Скалак)
    dip_ery = create_skalak_erythrocyte(R_typical)
    I_ery = adda_dda_laser(theta_vals, dip_ery)
    mn_e, mi_e, mx_e, ma_e = find_extrema(theta_vals, I_ery)
    ery_result = {
        'intensity': I_ery,
        'min_theta': mn_e, 'min_intensity': mi_e,
        'max_theta': mx_e, 'max_intensity': ma_e
    }
    print(f"Эритроцит: диполей={len(dip_ery)}")
    print(f"угол_min = {mn_e:.4f} рад ({mn_e*180/np.pi:.2f}°)")
    print(f"угол_max = {mx_e:.4f} рад ({mx_e*180/np.pi:.2f}°)")
    
    # ========== НАБОР ИЗ 3 ЭРИТРОЦИТОВ В ПЛОСКОСТИ XY ==========
    print("3 эритроцита в плоскости XY")
    centers_xy = [(-15, -15), (0, 0), (15, 15)]
    dip_ensemble = create_ensemble_in_plane(centers_xy, z_layer=0)
    I_ensemble = adda_dda_laser(theta_vals, dip_ensemble)
    mn_en, mi_en, mx_en, ma_en = find_extrema(theta_vals, I_ensemble)
    ensemble_result = {
        'intensity': I_ensemble,
        'min_theta': mn_en, 'min_intensity': mi_en,
        'max_theta': mx_en, 'max_intensity': ma_en
    }
    print(f"Диполей: {len(dip_ensemble)}")
    print(f"угол_min = {mn_en:.4f} рад ({mn_en*180/np.pi:.2f}°)")
    print(f"угол_max = {mx_en:.4f} рад ({mx_en*180/np.pi:.2f}°)")
    
    # Теория Эйри
    I_airy = airy_diffraction(theta_vals, R_typical)
    
    # 3D визуализация
    plot_3d_objects()
    
    # Дифракционные картины (с добавленным ансамблем)
    plot_diffraction_patterns(theta_vals, sphere_result, disk_results, biconcave_results,
                              long_result, short_result, ery_result, ensemble_result, I_airy)
    
    # Сравнение с теорией Эйри
    plot_comparison_airy(theta_vals, sphere_result, long_result, short_result, I_airy)



    plot_erythrocytes_in_xy_plane()
    

if __name__ == "__main__":
    main()