import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit, prange
from tqdm import tqdm

# --- TOUT LE CODE DE CALCUL RESTE IDENTIQUE ---
# ... (paramètres, fonctions numba, etc.) ...
# --- PARAMÈTRES (inchangés) ---
c = 1.0
n0 = 0.01
Lx, Ly = 400.0, 100.0
nx, ny = 800, 200
dx, dy = Lx / nx, Ly / ny
dt = 0.5
nt = 700 # Nombre de frames à générer pour la vidéo
total_time = nt * dt
t_max = (Lx / c) + 5 * 10.0
E0 = 5.0
tau = 10.0
k_laser = 1.0
N_electrons = 200000

# --- GRILLE ET PARTICULES (inchangés) ---
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X, Y = np.meshgrid(x, y)
np.random.seed(42)
positions = np.zeros((N_electrons, 2), dtype=np.float64)
positions[:, 0] = np.random.uniform(0, Lx, N_electrons)
positions[:, 1] = np.random.uniform(0, Ly, N_electrons)
velocities = np.zeros((N_electrons, 2), dtype=np.float64)
q_particle = n0 * Lx * Ly / N_electrons

# --- FONCTIONS NUMBA (inchangées) ---
@njit(parallel=True)
def compute_E_laser_cpu(x_grid, y_grid, t, E0, k_laser, c, tau, Ly, result, t_max):
    if t > t_max:
        result[:, :] = 0.0
        return
    for i in prange(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            envelope = E0 * math.exp(-((x_grid[i, j] - c * t) ** 2) / (2 * tau ** 2)) * \
                       math.exp(-((y_grid[i, j] - Ly / 2) ** 2) / (2 * (Ly / 10) ** 2))
            result[i, j] = envelope * math.cos(k_laser * (x_grid[i, j] - c * t))

@njit(parallel=True)
def initialize_rho_cpu(rho):
    for i in prange(rho.shape[0]):
        for j in range(rho.shape[1]):
            rho[i, j] = 0.0

@njit
def deposit_charge_cpu(positions, rho, q_particle, dx, dy, nx, ny):
    for i in range(positions.shape[0]):
        x_pos = positions[i, 0]
        y_pos = positions[i, 1]
        ix = int(x_pos / dx)
        iy = int(y_pos / dy)
        if 0 <= ix < nx and 0 <= iy < ny:
            rho[iy, ix] += q_particle / (dx * dy)

@njit(parallel=True)
def interpolate_E_fields_cpu(E_total_x, E_total_y, positions, dx, dy, nx, ny, E_interp_x, E_interp_y):
    for i in prange(positions.shape[0]):
        fx = positions[i, 0] / dx
        fy = positions[i, 1] / dy
        ix = int(fx) % nx
        iy = int(fy) % ny
        ix1 = (ix + 1) % nx
        iy1 = (iy + 1) % ny
        dx_f = fx - ix
        dy_f = fy - iy
        E_interp_x[i] = (1 - dx_f) * (1 - dy_f) * E_total_x[iy, ix] + \
                        dx_f * (1 - dy_f) * E_total_x[iy, ix1] + \
                        (1 - dx_f) * dy_f * E_total_x[iy1, ix] + \
                        dx_f * dy_f * E_total_x[iy1, ix1]
        E_interp_y[i] = (1 - dx_f) * (1 - dy_f) * E_total_y[iy, ix] + \
                        dx_f * (1 - dy_f) * E_total_y[iy, ix1] + \
                        (1 - dx_f) * dy_f * E_total_y[iy1, ix] + \
                        dx_f * dy_f * E_total_y[iy1, ix1]

@njit(parallel=True)
def update_particles_cpu(positions, velocities, E_interp_x, E_interp_y, dt, Lx, Ly):
    for i in prange(positions.shape[0]):
        c_val = 1.0
        v_x = velocities[i, 0]
        v_y = velocities[i, 1]
        v_sq = v_x**2 + v_y**2
        if v_sq >= c_val**2:
            v_sq = (c_val * 0.999999)**2
        gamma = 1.0 / math.sqrt(1.0 - v_sq / c_val**2)
        p_x = gamma * v_x + E_interp_x[i] * dt
        p_y = gamma * v_y + E_interp_y[i] * dt
        p_mag_sq = p_x**2 + p_y**2
        gamma_new = math.sqrt(1.0 + p_mag_sq / (c_val**2))
        velocities[i, 0] = p_x / gamma_new
        velocities[i, 1] = p_y / gamma_new
        positions[i, 0] = (positions[i, 0] + velocities[i, 0] * dt) % Lx
        positions[i, 1] = (positions[i, 1] + velocities[i, 1] * dt) % Ly

def solve_poisson_cpu_fft(rho, kx_sq_ky_sq):
    rho_k = np.fft.fft2(rho)
    phi_k = rho_k / kx_sq_ky_sq
    phi_k[0, 0] = 0.0
    phi = np.fft.ifft2(phi_k).real
    return phi

kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)
K_squared = KX**2 + KY**2
K_squared[0, 0] = 1.0

E_laser_grid = np.zeros((ny, nx), dtype=np.float64)
rho_grid = np.zeros((ny, nx), dtype=np.float64)
E_interp_x = np.zeros(N_electrons, dtype=np.float64)
E_interp_y = np.zeros(N_electrons, dtype=np.float64)
energies = np.zeros(N_electrons, dtype=np.float64)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

scatter = ax1.scatter([], [], s=0.5, color='white', alpha=1.0)
intensity = ax1.imshow(np.zeros((ny, nx)), extent=(0, Lx, 0, Ly), origin='lower',
                       cmap='inferno', alpha=1.0, vmin=0, vmax=E0**2)
ax1.set_xlim(0, Lx)
ax1.set_ylim(0, Ly)
ax1.set_xlabel('Position x', color='white')
ax1.set_ylabel('Position y', color='white')

bins = 100
hist_range = (0, 500)
ax2.set_xlim(hist_range)
ax2.set_xlabel(r'Énergie cinétique relativiste ($m_e c^2 (\gamma - 1)$)', color='white')
ax2.set_ylabel("Nombre d'électrons", color='white')
ax2.set_title('Distribution en Énergie des Électrons', color='white')

fig.patch.set_facecolor('black')
ax1.set_facecolor('black')
ax2.set_facecolor('black')
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

cbar = fig.colorbar(intensity, ax=ax1, label=r'Intensité EM ($E^2$)')
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.set_ylabel(r'Intensité EM ($E^2$)', color='white')
cbar.ax.set_facecolor('black')

progress_bar = tqdm(total=nt, desc="Génération de la vidéo", unit="frame")

def animate(i):
    t = i * dt
    compute_E_laser_cpu(X, Y, t, E0, k_laser, c, tau, Ly, E_laser_grid, t_max)
    initialize_rho_cpu(rho_grid)
    deposit_charge_cpu(positions, rho_grid, q_particle, dx, dy, nx, ny)
    phi = solve_poisson_cpu_fft(rho_grid - n0, K_squared)
    E_plasma_y, E_plasma_x = np.gradient(phi, dy, dx)
    E_total_x = -E_plasma_x + E_laser_grid
    E_total_y = -E_plasma_y
    interpolate_E_fields_cpu(E_total_x, E_total_y, positions, dx, dy, nx, ny, E_interp_x, E_interp_y)
    update_particles_cpu(positions, velocities, E_interp_x, E_interp_y, dt, Lx, Ly)

    scatter.set_offsets(positions[::4])
    intensity.set_data(E_laser_grid**2)

    v_sq = velocities[:, 0]**2 + velocities[:, 1]**2
    v_sq = np.minimum(v_sq, c**2 * 0.999999)
    gamma = 1.0 / np.sqrt(1.0 - v_sq / c**2)
    m_e = 1.0  # Ajout de la définition de la masse de l'électron (unité normalisée)
    energies = (gamma - 1.0) * (m_e * c**2)

    ax2.cla()
    hist, bin_edges = np.histogram(energies, bins=bins, range=hist_range)
    ax2.bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]), color='cyan', alpha=0.8)
    ax2.set_xlim(hist_range)
    ax2.set_yscale('log')
    if hist.max() > 0:
        ax2.set_ylim(0.5, hist.max() * 2)
    ax2.set_xlabel(r'Énergie cinétique relativiste ($m_e c^2 (\gamma - 1)$)', color='white')
    ax2.set_ylabel("Nombre d'électrons (échelle log)", color='white')
    ax2.set_title('Distribution en Énergie des Électrons', color='white')
    ax2.set_facecolor('black')

    ax1.set_title(f'Simulation LWFA - Temps: {t:.2f}', color='white')
    progress_bar.update(1)
    return scatter, intensity

# --- CRÉATION DE L'ANIMATION ET SAUVEGARDE ---

ani = animation.FuncAnimation(fig, animate, frames=nt, blit=False, repeat=False)

# ---- MODIFICATION IMPORTANTE ----
# Spécifier le chemin vers l'exécutable ffmpeg
# Remplacez '/usr/bin/ffmpeg' par le résultat de votre commande 'which ffmpeg'
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
# --------------------------------

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=2500) # Augmenté le bitrate pour une meilleure qualité

try:
    ani.save('lwfa_simulation.mp4', writer=writer, dpi=150)
    progress_bar.close()
    print("\nVidéo 'lwfa_simulation.mp4' sauvegardée avec succès !")
except Exception as e:
    progress_bar.close()
    print(f"\nUne erreur est survenue lors de la sauvegarde : {e}")
    print("Vérifiez que le chemin vers ffmpeg est correct et que vous avez les permissions d'écriture dans ce dossier.")
