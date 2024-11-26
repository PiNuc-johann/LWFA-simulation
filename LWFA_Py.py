import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import cuda, float64, int32

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
# Assure-toi que t_max <= total_time pour que le laser cesse avant la fin
t_max = (Lx / c) + 5 * 10.0  # Par exemple, 5 fois la durée de l'impulsion après le passage du laser

# Grille spatiale
x = np.linspace(0, Lx, nx, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
X_cpu, Y_cpu = np.meshgrid(x, y)

# Paramètres du laser ajustés
E0 = 5.0
tau = 10.0
k_laser = 1.0

# Nombre d'électrons augmenté
N_electrons = 20000

# Initialisation des positions et vitesses des électrons
np.random.seed(42)  # Pour la reproductibilité
positions = np.zeros((N_electrons, 2))
positions[:, 0] = np.random.uniform(0, Lx, N_electrons)
positions[:, 1] = np.random.uniform(0, Ly, N_electrons)
velocities = np.zeros((N_electrons, 2))

# Calcul de la charge par particule pour correspondre à la densité n0
q_particle = n0 * Lx * Ly / N_electrons

# Conversion des tableaux en arrays GPU
positions_device = cuda.to_device(positions)
velocities_device = cuda.to_device(velocities)
X_device = cuda.to_device(X_cpu)
Y_device = cuda.to_device(Y_cpu)

# Fonction du champ électrique laser avec profil gaussien en y
@cuda.jit
def compute_E_laser_gpu(x, y, t, E0, k_laser, c, tau, Ly, result, t_max):
    i, j = cuda.grid(2)
    if i < x.shape[0] and j < x.shape[1]:
        if t <= t_max:
            envelope = E0 * math.exp(-((x[i, j] - c * t) ** 2) / (2 * tau ** 2)) * \
                       math.exp(-((y[i, j] - Ly / 2) ** 2) / (2 * (Ly / 10) ** 2))
            result[i, j] = envelope * math.cos(k_laser * (x[i, j] - c * t))
        else:
            result[i, j] = 0.0  # Pas de champ laser après t_max

# Kernel pour initialiser rho_device à zéro
@cuda.jit
def initialize_rho_gpu(rho):
    i, j = cuda.grid(2)
    if i < rho.shape[0] and j < rho.shape[1]:
        rho[i, j] = 0.0

# Déposition de la charge sur la grille
@cuda.jit
def deposit_charge_gpu(positions, rho, q_particle, dx, dy, nx, ny):
    i = cuda.grid(1)
    if i < positions.shape[0]:
        x_pos = positions[i, 0]
        y_pos = positions[i, 1]
        ix = int(x_pos / dx) % nx
        iy = int(y_pos / dy) % ny
        cuda.atomic.add(rho, (iy, ix), q_particle / (dx * dy))

# Interpolation des champs électriques
@cuda.jit
def interpolate_E_fields_gpu(E_field_x, E_field_y, positions, dx, dy, nx, ny, E_interp_x, E_interp_y):
    i = cuda.grid(1)
    if i < positions.shape[0]:
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
@cuda.jit
def update_particles_gpu(positions, velocities, E_total_x, E_total_y, dt, Lx, Ly):
    i = cuda.grid(1)
    if i < positions.shape[0]:
        c_val = 1.0
        v_x = velocities[i, 0]
        v_y = velocities[i, 1]
        v_sq = v_x**2 + v_y**2
        gamma = 1.0 / math.sqrt(1 - v_sq / c_val**2)

        # Mise à jour du moment
        p_x = gamma * v_x + E_total_x[i] * dt
        p_y = gamma * v_y + E_total_y[i] * dt
        gamma_new = math.sqrt(1 + p_x**2 + p_y**2)

        # Mise à jour des vitesses
        velocities[i, 0] = p_x / gamma_new
        velocities[i, 1] = p_y / gamma_new

        # Mise à jour des positions
        positions[i, 0] = (positions[i, 0] + velocities[i, 0] * dt) % Lx
        positions[i, 1] = (positions[i, 1] + velocities[i, 1] * dt) % Ly

# Fonction pour résoudre l'équation de Poisson sur le CPU
def solve_poisson_cpu(rho):
    # Transformée de Fourier
    rho_k = np.fft.fft2(rho)

    # Vecteurs k
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_squared = KX**2 + KY**2
    K_squared[0, 0] = 1.0  # Pour éviter la division par zéro

    # Calcul du potentiel
    phi_k = rho_k / K_squared
    phi_k[0, 0] = 0.0  # Le potentiel moyen peut être fixé à zéro

    # Transformée de Fourier inverse
    phi = np.fft.ifft2(phi_k).real

    return phi

# Pré-allocation des tableaux GPU nécessaires
E_l_grid_device = cuda.device_array((ny, nx), dtype=np.float64)
rho_device = cuda.device_array((ny, nx), dtype=np.float64)
E_field_x_device = cuda.device_array((ny, nx), dtype=np.float64)
E_field_y_device = cuda.device_array((ny, nx), dtype=np.float64)
E_interp_x_device = cuda.device_array(N_electrons, dtype=np.float64)
E_interp_y_device = cuda.device_array(N_electrons, dtype=np.float64)

# Pré-allocation pour l'énergie des électrons (CPU)
energies = np.zeros(N_electrons, dtype=np.float64)

# Préparation de l'animation avec deux subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

# Subplot 1: Positions des électrons et intensité du champ laser
scatter = ax1.scatter([], [], s=0.5, color='white', alpha=1.0)
intensity = ax1.imshow(np.zeros((ny, nx)), extent=(0, Lx, 0, Ly), origin='lower',
                       cmap='inferno', alpha=1.0, vmin=0, vmax=E0**2)
ax1.set_xlim(0, Lx)
ax1.set_ylim(0, Ly)
ax1.set_xlabel('Position x', color='white')
ax1.set_ylabel('Position y', color='white')
ax1.set_title('Simulation 2D LWFA: Formation de Bulles et Accélération des Électrons', color='white')

# Subplot 2: Distribution en énergie des électrons
bins = 50  # Nombre de bins pour l'histogramme
hist, bin_edges = np.histogram(energies, bins=bins, range=(0, 50))
hist_plot = ax2.bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]), color='white', alpha=0.7)
ax2.set_xlim(0, 50)
ax2.set_ylim(0, N_electrons / bins)
ax2.set_xlabel('Énergie relativiste (γ - 1)', color='white')
ax2.set_ylabel('Nombre d\'électrons', color='white')
ax2.set_title('Distribution en Énergie des Électrons', color='white')

# Configuration des couleurs des ticks et labels
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

# Configuration de la colorbar avec les labels en blanc et fond noir
cbar = fig.colorbar(intensity, ax=ax1, label='Intensité EM (proportionnelle à $E^2$)')
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.set_ylabel('Intensité EM (proportionnelle à $E^2$)', color='white')
cbar.ax.set_facecolor('black')

# Assurer que toute la figure a un fond noir
fig.patch.set_facecolor('black')
ax1.set_facecolor('black')
ax2.set_facecolor('black')

def animate(i):
    global positions_device, velocities_device, E_l_grid_device, rho_device, E_field_x_device, E_field_y_device, E_interp_x_device, E_interp_y_device, energies

    t = i * dt

    # Calcul du champ laser sur GPU avec condition t <= t_max
    threadsperblock_laser = (16, 16)
    blockspergrid_x_laser = (ny + threadsperblock_laser[0] - 1) // threadsperblock_laser[0]
    blockspergrid_y_laser = (nx + threadsperblock_laser[1] - 1) // threadsperblock_laser[1]
    blockspergrid_laser = (blockspergrid_x_laser, blockspergrid_y_laser)

    compute_E_laser_gpu[blockspergrid_laser, threadsperblock_laser](X_device, Y_device, t, E0, k_laser, c, tau, Ly, E_l_grid_device, t_max)

    # Initialisation de rho_device à zéro
    threadsperblock_init = (16, 16)
    blockspergrid_x_init = (ny + threadsperblock_init[0] - 1) // threadsperblock_init[0]
    blockspergrid_y_init = (nx + threadsperblock_init[1] - 1) // threadsperblock_init[1]
    blockspergrid_init = (blockspergrid_x_init, blockspergrid_y_init)
    initialize_rho_gpu[blockspergrid_init, threadsperblock_init](rho_device)

    # Calcul de la densité de charge sur GPU
    threadsperblock_charge = 256
    blockspergrid_charge = (N_electrons + threadsperblock_charge - 1) // threadsperblock_charge
    deposit_charge_gpu[blockspergrid_charge, threadsperblock_charge](positions_device, rho_device, q_particle, dx, dy, nx, ny)

    # Récupération de rho depuis le GPU pour résoudre Poisson sur le CPU
    rho_host = rho_device.copy_to_host()
    phi = solve_poisson_cpu(rho_host)

    # Calcul des champs électriques sur le CPU
    E_field_x = -np.gradient(phi, dx, axis=1)
    E_field_y = -np.gradient(phi, dy, axis=0)

    # Récupération de E_l_grid depuis le GPU
    E_l_grid_host = E_l_grid_device.copy_to_host()

    # Calcul du champ total
    E_total_x = E_field_x + E_l_grid_host  # Ajouter le champ laser
    E_total_y = E_field_y

    # Assignation correcte des champs électriques au GPU
    cuda.to_device(E_total_x, to=E_field_x_device)
    cuda.to_device(E_total_y, to=E_field_y_device)

    # Interpolation des champs sur GPU
    threadsperblock_interp = 256
    blockspergrid_interp = (N_electrons + threadsperblock_interp - 1) // threadsperblock_interp
    interpolate_E_fields_gpu[blockspergrid_interp, threadsperblock_interp](
        E_field_x_device, E_field_y_device, positions_device, dx, dy, nx, ny, E_interp_x_device, E_interp_y_device)

    # Mise à jour des particules sur GPU
    update_particles_gpu[blockspergrid_charge, threadsperblock_charge](
        positions_device, velocities_device, E_interp_x_device, E_interp_y_device, dt, Lx, Ly)

    # Récupération des positions pour l'affichage
    positions_host = positions_device.copy_to_host()
    scatter.set_offsets(positions_host)

    # Récupération et mise à jour de l'intensité du champ laser
    E_l_grid = E_l_grid_device.copy_to_host()
    intensity_field = E_l_grid**2
    intensity.set_data(intensity_field)

    # Récupération des vitesses pour calculer l'énergie
    velocities_host = velocities_device.copy_to_host()
    # Calcul de l'énergie relativiste γ - 1
    v_sq = velocities_host[:, 0]**2 + velocities_host[:, 1]**2
    gamma = 1.0 / np.sqrt(1 - v_sq / c**2)
    energies = gamma - 1.0

    # Mise à jour de l'histogramme de distribution en énergie
    ax2.cla()  # Effacer l'histogramme précédent
    hist, bin_edges = np.histogram(energies, bins=bins, range=(0, 10000))
    ax2.bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]), color='white', alpha=0.7)
    ax2.set_xlim(0, 10000)
    ax2.set_yscale('log')
    ax2.set_ylim(0, np.max(hist) * 1.1)
    ax2.set_xlabel('Énergie relativiste (γ - 1)', color='white')
    ax2.set_ylabel('Nombre d\'électrons', color='white')
    ax2.set_title('Distribution en Énergie des Électrons', color='white')
    ax2.set_facecolor('black')

    # Mise à jour du titre avec la couleur blanche
    ax1.set_title(f'Simulation 2D LWFA: t={t:.2f}', color='white')
    
    return scatter, intensity, ax2.patches

# Configuration de l'enregistreur vidéo
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30*2,  # fps=30 et speed_factor=2
                metadata=dict(artist='Me'),
                bitrate=2000)

# Créer l'animation avec blit=False pour éviter l'erreur liée à '_resize_id' et repeat=False pour ne pas répéter l'animation
ani = animation.FuncAnimation(fig, animate, frames=nt, interval=1000/60, blit=False, repeat=False)


# Afficher l'animation
plt.show()
