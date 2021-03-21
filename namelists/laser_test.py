import tempfile

from custom_lasers import ChirpedLaser
from scipy.constants import c, e, m_e, pi

# Laser properties
wavelength_si = 800.e-9 # m
fwhm_si = 30.e-15 # s
energy_si = 100.e-3 # J
focal_spot_si = 3.e-6 # m

# Plasma properties
energy_max_mev = 30e6 # eV
n_0 = 2 # Used to compute simulation resolution, no species will have this density

# Simulation Properties
box_front_si = 15e-6
box_back_si = 15e-6
particles_per_cell = 64
boundary_conditions = [["remove", "remove"]]
simulation_time_si = 150.e-15 # s
number_of_patches = [32]

# Computed Laser Properties
omega_si = 2 * pi * c / wavelength_si # rad s^(-1)
intensity_si = energy_si / (fwhm_si * pi * focal_spot_si**2)
beta = [0., 0., 0., 0., 0.]
a_0 = 0.85 * np.sqrt(intensity_si / 1.e22 * (wavelength_si / 1.e-6)**2)
tau_si = fwhm_si / (2 * np.sqrt(np.log(2)))
laser = ChirpedLaser(omega_si, tau_si, a_0, beta)

# Computed Plasma Properties
density = 2.
thickness_si = 0.e-6
thickness = thickness_si * omega_si / c
energy_max_si = energy_max_mev * e
energy_max = energy_max_si / (m_e * c**2)
momentum_max_si = np.sqrt(2 * 1836. * m_e * energy_max_si)
momentum_max = momentum_max_si / (m_e * c)
lambda_p = np.sqrt(n_0)

# Computed Simulation Properties
cell_length = [0.05 * lambda_p] # calculate plasma wavelength and resolve much smaller (lambda_p / 20)
box_front = box_back_si * omega_si / c
box_back = box_back_si * omega_si / c
number_of_cells = [np.ceil(( (box_front + thickness + box_back) / cell_length[0]) / number_of_patches[0]) * number_of_patches[0]]
screen_position = box_front + thickness + box_back
timestep = 0.99 * cell_length[0]
number_of_timesteps = int(np.ceil(((simulation_time_si + laser.get_peak_offset()) * omega_si) / timestep))
diag_every = 1

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
    reference_angular_frequency_SI = omega_si,
    random_seed = 0,
    print_every = number_of_timesteps / 10
)

LoadBalancing(
    initial_balance = True,
    every = number_of_timesteps / 100
)

Laser(
    box_side = "xmin",
    space_time_profile = [ lambda t: 0., lambda t: laser.at_sim_time(t) ]
)

LaserPlanar1D(
    box_side = "xmax",
    a0 = a_0,
    omega = 1.,
    polarization_phi = 0.,
    ellipticity = 0.,
    time_envelope = tgaussian(fwhm=fwhm_si*omega_si, start=0., center=laser.get_peak_offset()*omega_si)
)

DiagScalar(every=diag_every)

DiagFields(every=diag_every)
