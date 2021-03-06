import numpy as np

from numpy import ceil, floor, log, sqrt
from scipy.constants import c, m_e, pi

# Laser properties
wavelength_si = 800.e-9 # m
fwhm_si = 30.e-15 # s
energy_si = 100.e-3 # J
focal_spot_si = 3.e-6 # m
omega_si = 2 * pi * c / wavelength_si # rad s^(-1)

intensity = energy_si / (fwhm_si * pi * focal_spot_si**2)
a_0 = 0.85 * sqrt(intensity / 1.e22 * (wavelength_si / 1.e-6)**2)
fwhm = fwhm_si * omega_si
laser_t0 = fwhm / sqrt(log(2)) # time when peak of laser enters box

# Plasma properties
n_0 = 2. # relative to n_crit
lambda_p = sqrt(n_0)
cell_length = [0.05 * lambda_p] # calculate plasma wavelength and resolve much smaller (lambda_p / 50)
number_of_patches = [32]
particles_per_cell = 64
thickness_si = 5.e-6 # m
thickness = thickness_si * omega_si / c
number_of_cells = [ceil((8 * thickness / cell_length[0]) / number_of_patches[0]) * number_of_patches[0]]
boundary_conditions = [["remove", "remove"]]

# Simulation properties
simulation_time_si = 1e-12 # s
cfl_condition = cell_length[0] # 1 / sqrt(sum([1./l**2 for l in Main.cell_length]))
timestep = 0.99 * cfl_condition
number_of_timesteps = int(ceil((simulation_time_si * omega_si) / timestep))
diag_every = number_of_timesteps

try:
    with open("par_vec.npy", 'rb') as d_map_file:
        density_map = np.load(d_map_file, allow_pickle=False)
except FileNotFoundError:
    density_map = [0.]

Main(
    geometry = "1Dcartesian",
    interpolation_order = 2,
    number_of_cells = number_of_cells,
    cell_length = cell_length,
    number_of_timesteps = number_of_timesteps,
    timestep = timestep,
    number_of_patches = number_of_patches,
    patch_arrangement = "hilbertian",
    maxwell_solver = "Yee",
    EM_boundary_conditions = [["silver-muller", "silver-muller"],],
    time_fields_frozen = 0.,
    reference_angular_frequency_SI = omega_si, # Probably not necessary as we're not doing anything with collisions/ionisation
    random_seed = 0,
    print_every = number_of_timesteps / 10
)

LoadBalancing(
    initial_balance = True,
    every = number_of_timesteps / 100
)

LaserPlanar1D(
    box_side = "xmin",
    a0 = a_0,
    omega = 1.,
    polarization_phi = 0.,
    ellipticity = 0.,
    time_envelope = tgaussian(fwhm=fwhm, start=0., center=laser_t0)
)

def number_density(x):
    if (rel := x - thickness * 3.5) >= 0 and rel < thickness:
        return density_map[int(floor(rel * len(density_map) / thickness))]
    else:
        return 0.

Species(
    name = "electrons",
    position_initialization = 'regular',
    momentum_initialization = 'cold',
    ionization_model = 'none',
    particles_per_cell = particles_per_cell,
    mass = 1.,
    charge = -1.,
    number_density = number_density,
    boundary_conditions = boundary_conditions
)

Species(
    name = "protons",
    position_initialization = 'regular',
    momentum_initialization = 'cold',
    ionization_model = 'none',
    particles_per_cell = particles_per_cell,
    mass = 1836.,
    charge = 1.,
    number_density = number_density,
    boundary_conditions = boundary_conditions
)

DiagScreen(
    shape = "plane",
    point = [8 * thickness],
    vector = [1.],
    direction = "forward",
    deposited_quantity = "weight_ekin",
    species = ["protons"],
    axes = [],
    every = diag_every
)
