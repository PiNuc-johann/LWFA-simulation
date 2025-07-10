import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit, prange # Importation de jit et prange pour le CPU

# Paramètres physiques ajustés
c = 1.0
e_charge = 1.0
m_e = 1.0
epsilon_0 = 1.0
n0 = 0.01

# Paramètres de simulation
Lx, Ly = 400.0, 100.0
nx, ny = 800, 200
dx, dy = Lx / nx, Ly / ny
dt = 0.5
nt = 1000  # Nombre de frames
total_time = nt * dt  # Temps total de la simulation

# Définir t_max pour arrêter l'envoi du laser après t_max
t_max = (Lx / c) + 5 * 10.0

# Grille spatiale
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X, Y = np.meshgrid(x, y)

# Paramètres du laser ajustés
E0 = 5.0
tau = 10.0
k_laser = 1.0

# Nombre d'électrons augmenté
N_electrons = 20000

# Initialisation des positions et vitesses des électrons (tableaux NumPy standard)
np.random.seed(42)
positions = np.zeros((N_electrons, 2), dtype=np.float64)
positions[:, 0] = np.random.uniform(0, Lx, N_electrons)
positions[:, 1] = np.random.uniform(0, Ly, N_electrons)
velocities = np.zeros((N_electrons, 2), dtype=np.float64)

# Calcul de la charge par particule
q_particle = n0 * Lx * Ly / N_electrons

# --- Fonctions Numba pour le CPU ---

# Fonction du champ électrique laser avec profil gaussien en y
@jit(nopython=True, parallel=True)
def compute_E_laser_cpu(x_grid, y_grid, t, E0, k_laser, c, tau, Ly, result, t_max):
    if t > t_max:
        result[:, :] = 0.0
        return

    # Boucles parallèles sur la grille
    for i in prange(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            envelope = E0 * math.exp(-((x_grid[i, j] - c * t) ** 2) / (2 * tau ** 2)) * \
                       math.exp(-((y_grid[i, j] - Ly / 2) ** 2) / (2 * (Ly / 10) ** 2))
            result[i, j] = envelope * math.cos(k_laser * (x_grid[i, j] - c * t))

# Déposition de la charge sur la grille (NON parallèle pour éviter les race conditions)
@jit(nopython=True)
def deposit_charge_cpu(positions, rho, q_particle, dx, dy, nx, ny):
    # La grille rho doit être initialisée à zéro avant d'appeler cette fonction
    for i in range(positions.shape[0]):
        x_pos = positions[i, 0]
        y_pos = positions[i, 1]
        ix = int(x_pos / dx)
        iy = int(y_pos / dy)
        # S'assurer que les indices restent dans les bornes (avec le modulo)
        if 0 <= ix < nx and 0 <= iy < ny:
            rho[iy, ix] += q_particle / (dx * dy)

# Interpolation des champs électriques
@jit(nopython=True, parallel=True)
def interpolate_E_fields_cpu(E_field_x, E_field_y, positions, dx, dy, nx, ny, E_interp_x, E_interp_y):
    for i in prange(positions.shape[0]):
        # Calcul des indices et fractions
        fx = positions[i, 0] / dx
        fy = positions[i, 1] / dy
        ix = int(fx) % nx
        iy = int(fy) % ny
        ix1 = (ix + 1) % nx
        iy1 = (iy + 1) % ny
        dx_f = fx - ix
        dy_f = fy - iy

        # Interpolation bilinéaire
        E_interp_x[i] = (1 - dx_f) * (1 - dy_f) * E_field_x[iy, ix] + \
                        dx_f * (1 - dy_f) * E_field_x[iy, ix1] + \
                        (1 - dx_f) * dy_f * E_field_x[iy1, ix] + \
                        dx_f * dy_f * E_field_x[iy1, ix1]

        E_interp_y[i] = (1 - dx_f) * (1 - dy_f) * E_field_y[iy, ix] + \
                        dx_f * (1 - dy_f) * E_field_y[iy, ix1] + \
                        (1 - dx_f) * dy_f * E_field_y[iy1, ix] + \
                        dx_f * dy_f * E_field_y[iy1, ix1]

# Mise à jour des particules
@jit(nopython=True, parallel=True)
def update_particles_cpu(positions, velocities, E_total_x, E_total_y, dt, Lx, Ly):
    c_val = 1.0
    for i in prange(positions.shape[0]):
        v_x = velocities[i, 0]
        v_y = velocities[i, 1]
        v_sq = v_x**2 + v_y**2

        # S'assurer que v < c pour éviter les erreurs de domaine
        if v_sq >= c_val**2:
            v_sq = c_val**2 * 0.999999

        gamma = 1.0 / math.sqrt(1 - v_sq / c_val**2)

        # Mise à jour du moment
        p_x = gamma * v_x + E_total_x[i] * dt
        p_y = gamma * v_y + E_total_y[i] * dt
        gamma_new = math.sqrt(1 + p_x**2 + p_y**2)

        # Mise à jour des vitesses
        velocities[i, 0] = p_x / gamma_new
        velocities[i, 1] = p_y / gamma_new

        # Mise à jour des positions avec conditions aux limites périodiques
        positions[i, 0] = (positions[i, 0] + velocities[i, 0] * dt) % Lx
        positions[i, 1] = (positions[i, 1] + velocities[i, 1] * dt) % Ly

# Fonction pour résoudre l'équation de Poisson sur le CPU (inchangée)
def solve_poisson_cpu(rho):
    rho_k = np.fft.fft2(rho)
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_squared = KX**2 + KY**2
    K_squared[0, 0] = 1.0
    phi_k = rho_k / K_squared
    phi_k[0, 0] = 0.0
    phi = np.fft.ifft2(phi_k).real
    return phi

# Pré-allocation des tableaux NumPy nécessaires
E_l_grid = np.zeros((ny, nx), dtype=np.float64)
rho = np.zeros((ny, nx), dtype=np.float64)
E_interp_x = np.zeros(N_electrons, dtype=np.float64)
E_interp_y = np.zeros(N_electrons, dtype=np.float64)
energies = np.zeros(N_electrons, dtype=np.float64)

# --- Préparation de l'animation ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

scatter = ax1.scatter([], [], s=0.5, color='white', alpha=1.0)
intensity = ax1.imshow(np.zeros((ny, nx)), extent=(0, Lx, 0, Ly), origin='lower',
                       cmap='inferno', alpha=1.0, vmin=0, vmax=E0**2)
ax1.set_xlim(0, Lx)
ax1.set_ylim(0, Ly)
ax1.set_xlabel('Position x', color='white')
ax1.set_ylabel('Position y', color='white')

bins = 50
hist, bin_edges = np.histogram(energies, bins=bins, range=(0, 50))
ax2.bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]), color='white', alpha=0.7)
ax2.set_xlim(0, 50)
ax2.set_ylim(0, N_electrons / bins)
ax2.set_xlabel('Énergie relativiste (γ - 1)', color='white')
ax2.set_ylabel('Nombre d\'électrons', color='white')
ax2.set_title('Distribution en Énergie des Électrons', color='white')

plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
cbar = fig.colorbar(intensity, ax=ax1, label='Intensité EM (proportionnelle à $E^2$)')
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.set_ylabel('Intensité EM (proportionnelle à $E^2$)', color='white')

fig.patch.set_facecolor('black')
ax1.set_facecolor('black')
ax2.set_facecolor('black')

# --- Boucle d'animation ---
def animate(i):
    global positions, velocities, E_l_grid, rho, E_interp_x, E_interp_y, energies

    t = i * dt

    # 1. Calculer le champ laser
    compute_E_laser_cpu(X, Y, t, E0, k_laser, c, tau, Ly, E_l_grid, t_max)

    # 2. Déposer la charge des particules sur la grille
    rho.fill(0.0) # Réinitialiser la grille de densité
    deposit_charge_cpu(positions, rho, q_particle, dx, dy, nx, ny)

    # 3. Résoudre l'équation de Poisson pour obtenir le potentiel phi
    phi = solve_poisson_cpu(rho)

    # 4. Calculer les champs électriques à partir du potentiel
    E_plasma_x = -np.gradient(phi, dx, axis=1)
    E_plasma_y = -np.gradient(phi, dy, axis=0)

    # 5. Champ total (plasma + laser) sur la grille
    E_total_grid_x = E_plasma_x + E_l_grid
    E_total_grid_y = E_plasma_y

    # 6. Interpoler les champs totaux aux positions des particules
    interpolate_E_fields_cpu(
        E_total_grid_x, E_total_grid_y, positions, dx, dy, nx, ny, E_interp_x, E_interp_y)

    # 7. Mettre à jour les positions et vitesses des particules
    update_particles_cpu(
        positions, velocities, E_interp_x, E_interp_y, dt, Lx, Ly)

    # --- Mise à jour des graphiques ---
    scatter.set_offsets(positions)
    intensity.set_data(E_l_grid**2)

    # Calcul de l'énergie relativiste γ - 1
    v_sq = velocities[:, 0]**2 + velocities[:, 1]**2
    # Petite sécurité pour éviter que v >= c
    v_sq = np.minimum(v_sq, c**2 * 0.9999999)
    gamma = 1.0 / np.sqrt(1 - v_sq / c**2)
    energies = gamma - 1.0

    # Mise à jour de l'histogramme
    ax2.cla()
    hist, bin_edges = np.histogram(energies, bins=100, range=(0, 2500))
    ax2.bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]), color='white', alpha=0.7)
    ax2.set_xlim(0, 2500)
    ax2.set_yscale('log')
    ax2.set_ylim(1, N_electrons/10) # Ajuster les limites pour l'échelle log
    ax2.set_xlabel('Énergie relativiste (γ - 1)', color='white')
    ax2.set_ylabel("Nombre d'électrons (log)", color='white')
    ax2.set_title('Distribution en Énergie des Électrons', color='white')
    ax2.set_facecolor('black')

    ax1.set_title(f'Simulation 2D LWFA (CPU): t={t:.2f}', color='white')

    return scatter, intensity, *ax2.patches

# Créer et lancer l'animation
ani = animation.FuncAnimation(fig, animate, frames=nt, interval=20, blit=False, repeat=False)
plt.show()

# Pour sauvegarder l'animation (décommenter la ligne suivante)
# print("Sauvegarde de l'animation...")
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
# ani.save('lwfa_cpu_simulation.mp4', writer=writer)
# print("Sauvegarde terminée.")
