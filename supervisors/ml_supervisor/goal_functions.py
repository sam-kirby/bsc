import happi
import logging
import numpy as np

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
