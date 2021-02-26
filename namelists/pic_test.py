from math import pi
from numpy import ceil, log, sqrt
from scipy.constants import c, m_e

# Laser properties
wavelength_si = 800.e-9 # m
fwhm_si = 30.e-15 # s
energy_si = 100.e-3 # J
focal_spot_si = 6.e-6 # m
omega_si = 2 * pi * c / wavelength_si # rad s^(-1)

intensity = energy_si / (fwhm_si * pi * focal_spot_si**2)
a_0 = 0.85 * sqrt(intensity / 1.e22 * (wavelength_si / 1.e-6)**2)
fwhm = fwhm_si * omega_si
laser_t0 = fwhm/(sqrt(log(2))) # time when peak of laser enters box

# Plasma properties
n_0 = 5. # relative to n_crit
lambda_p = sqrt(n_0)
cell_length = [0.01 * lambda_p] # calculate plasma wavelength and resolve much smaller (lambda_p / 100)
number_of_patches = [256]
particles_per_cell = 100
thickness_si = 10.e-6 # m
thickness = thickness_si * omega_si / c
number_of_cells = [ceil((6 * thickness / cell_length[0]) / 256) * 256]
boundary_conditions = [["remove", "remove"]]
energy_max_p_mev = .25 # MeV
energy_max_p_si = energy_max_p_mev * 1.602e-13
momentum_max_p_si = sqrt(energy_max_p_si / (1836. * m_e)) * 1836. * m_e 
momentum_max_p = momentum_max_p_si / (m_e * c)

# Simulation properties
number_of_timesteps = 60000
timestep_over_cfl = 0.99
diag_every = 600

Main(
    geometry = "1Dcartesian",
    interpolation_order = 2,
    number_of_cells = number_of_cells,
    cell_length = cell_length,
    number_of_timesteps = number_of_timesteps,
    timestep_over_CFL = timestep_over_cfl,
    number_of_patches = number_of_patches,
    patch_arrangement = "hilbertian",
    maxwell_solver = "Yee",
    EM_boundary_conditions = [["silver-muller", "silver-muller"],],
    time_fields_frozen = 0.,
    reference_angular_frequency_SI = omega_si, # Probably not necessary as we're not doing anything with collisions/ionisation
    random_seed = 0,
    print_every = int(number_of_timesteps/10)
)

LaserPlanar1D(
    box_side = "xmin",
    a0 = a_0,
    omega = 1.,
    polarization_phi = 0.,
    ellipticity = 0.,
    time_envelope = tgaussian(fwhm=fwhm, start=0., center=laser_t0)
)

LoadBalancing(
    every = diag_every,
    initial_balance = False
)

number_density = trapezoidal(n_0, xvacuum=3*thickness, xplateau=thickness)
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

DiagScalar(every = diag_every)

DiagFields(every = diag_every)

DiagParticleBinning(
    every = diag_every,
    deposited_quantity = "weight",
    species = ["protons"],
    axes = [
        ["x", 0, number_of_cells[0] * cell_length[0], number_of_cells[0]],
        ["px", -momentum_max_p, momentum_max_p, 500, "edge_inclusive"]
    ]
)

DiagParticleBinning(
    every = diag_every,
    deposited_quantity = "weight",
    species = ["protons"],
    axes = [
        ["ekin", 0.02, 2.5, 100, "edge_inclusive"]
    ]
)
