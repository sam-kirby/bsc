from custom_lasers import ChirpedLaser
from scipy.constants import c, e, m_e, pi

# Laser properties
wavelength_si = 800.e-9 # m
fwhm_si = 30.e-15 # s
energy_si = 100.e-3 # J
focal_spot_si = 3.e-6 # m

# Plasma properties
energy_max_mev = 30e6 # eV
energy_bins = 50000
n_0 = 2 # Used to compute simulation resolution, no species will have this density

# Simulation Properties
box_front_si = 15e-6
box_back_si = 10e-6
particles_per_cell = 64
boundary_conditions = [["remove", "remove"]]
simulation_time_si = 1e-12 # s
number_of_patches = [32]

with open("par_vec.npy", "rb") as f:
    parameter_vec = np.load(f, allow_pickle=False)

# Computed Laser Properties
omega_si = 2 * pi * c / wavelength_si # rad s^(-1)
intensity_si = energy_si / (fwhm_si * pi * focal_spot_si**2)
beta = [0., 0.]
beta.extend(parameter_vec[0:3])
a_0 = 0.85 * np.sqrt(intensity_si / 1.e22 * (wavelength_si / 1.e-6)**2)
tau_si = fwhm_si / (2 * np.sqrt(np.log(2)))
laser = ChirpedLaser(omega_si, tau_si, a_0, beta)

# Computed Plasma Properties
density = parameter_vec[3]
thickness_si = parameter_vec[4]
thickness = thickness_si * omega_si / c
energy_max_si = energy_max_mev * e
energy_max = energy_max_si / (m_e * c**2)
momentum_max_si = np.sqrt(2 * 1836. * m_e * energy_max_si)
momentum_max = momentum_max_si / (m_e * c)
lambda_p = np.sqrt(n_0)

# Computed Simulation Properties
cell_length = [0.05 * lambda_p] # calculate plasma wavelength and resolve much smaller (lambda_p / 20)
box_front = box_front_si * omega_si / c
box_back = box_back_si * omega_si / c
number_of_cells = [np.ceil(((box_front + thickness + box_back) / cell_length[0]) / number_of_patches[0]) * number_of_patches[0]]
screen_position = box_front + thickness + box_back
timestep = 0.99 * cell_length[0]
number_of_timesteps = int(np.ceil(((simulation_time_si + laser.get_peak_offset()) * omega_si) / timestep))
diag_every = number_of_timesteps

results_dir = None

def preprocess():
    import os
    import tempfile
    global results_dir
    results_dir = os.getcwd()
    work_dir = tempfile.mkdtemp(dir=os.environ["TMPDIR"])
    os.chdir(work_dir)

def analysis():
    import os
    import pickle
    import shutil
    from h5py import File # use h5py to avoid reimporting the namelist and regenerating laser

    with File("Screen0.h5") as f:
        data = np.array(f[f"timestep{number_of_timesteps - 1:0>8d}"])

    try:
        index = np.nonzero(data)[0][-1]
    except IndexError:
        index = 0
    
    result = index / energy_bins * energy_max_mev

    with open(f"{results_dir}/result", "wb") as f:
        pickle.dump(-result, f)

    shutil.rmtree(os.getcwd())

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
    print_every = int(number_of_timesteps / 10)
)

LoadBalancing(
    initial_balance = True,
    every = number_of_timesteps / 100
)

Laser(
    box_side = "xmin",
    space_time_profile = [ lambda t: 0., lambda t: laser.at_sim_time(t) ]
)

number_density = trapezoidal(density, xvacuum=box_front, xplateau=thickness)

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
    point = [screen_position],
    vector = [1.],
    direction = "forward",
    deposited_quantity = "weight",
    species = ["protons"],
    axes = [
        ["ekin", 0., energy_max, energy_bins]
    ],
    every = number_of_timesteps - 1
)
