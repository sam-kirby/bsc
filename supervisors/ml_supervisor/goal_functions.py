import happi
import logging
import numpy as np
import pickle
import time

def max_energy_negated(work_dir: str) -> float:
    """
    Find the highest energy bin and return its energy * -1
    """
    logger = logging.getLogger("supervisor")
    sim_results = happi.Open(work_dir)
    logger.debug(f"simulation results opened in {work_dir}")
    final_energy_spectrum = sim_results.ParticleBinning(
        0,
        timesteps=sim_results.namelist.Main.number_of_timesteps
    )
    last_occupied_energy_bin = np.nonzero(final_energy_spectrum.getData()[0])[0][-1]
    if last_occupied_energy_bin == len(final_energy_spectrum.getData()[0]) - 1:
        logger.warning("Final energy bin not empty, data loss may have occurred")
    return -final_energy_spectrum._centers[0][last_occupied_energy_bin]

def screen_dep_energy_negated(work_dir: str) -> float:
    """
    Find the energy deposited on screen 0 and return * -1
    """
    logger = logging.getLogger("supervisor")
    sim_results = happi.Open(work_dir)
    logger.debug(f"simulation results opened in {work_dir}")
    return -sim_results.Screen(0).getData()[-1]

def load_result_from_file(work_dir: str) -> float:
    """
    Load a pickled result from the work dir - useful if analysis done by smilei
    """
    time.sleep(1)  # This wait is necessary to give Smilei time to shutdown
    with open(f"{work_dir}/result", "rb") as result_file:
        return pickle.load(result_file)
