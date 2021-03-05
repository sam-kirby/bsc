import logging
import numpy as np
import os
import pathlib
import shutil
import tempfile
import threading
import time

from collections.abc import Callable
from mpi4py import MPI


class SmileiWrapper:
    def __init__(
        self,
        namelist: pathlib.Path,
        post_process: Callable[[str], float],
        analysis_concurrency: int
    ):
        self.namelist = namelist
        self.post_process = post_process
        self.generation = None
        self.analysis_concurrency = analysis_concurrency
        self.mpi_spawn_lock = threading.Lock()
        self.analysis_semaphore = threading.Semaphore(analysis_concurrency)

    def run_sim(self, par_vec: np.ndarray) -> float:
        """
        Run a single Smilei simulation in another MPI process

        Note the simulation is run in a temporary directory. Any results needed should be either be 
        processed or copied in the post_process function

        Arguments:
        par_vec -- a parameter vector to pass to Smilei 
            - will be stored as the variable x and can be accessed in the namelist
        namelist -- path to the namelist
        post_process -- a function to use to process the results, taking the work directory as an
            argument
        
        Returns:
        The result of post_process
        """
        logger = logging.getLogger("supervisor")

        # create temporary work directory - have to do it this way as directory can fail to delete on HPC...
        work_dir = tempfile.mkdtemp(dir=os.getcwd())

        # set working directory for child process
        info = MPI.Info.Create()
        info.Set("wdir", work_dir)

        # spawn smilei child process
        with self.mpi_spawn_lock:
            logger.debug(f"Starting Smilei simulation with parameters: {', '.join([str(x) for x in par_vec])}")

            inter = MPI.COMM_SELF.Spawn(
                command = "smilei_sub",
                args=[
                    "from numpy import array",
                    "x = {}".format(par_vec.__repr__().replace("\n", "")),
                    bytes(self.namelist.resolve())
                ],
                maxprocs=1,
                info=info
            )

            logger.debug("Process spawned, waiting for completion")

        # wait for smilei to finish (yielding to other threads)
        req = inter.Ibarrier()
        while not req.Test():
            time.sleep(5)

        inter.Disconnect()

        # perform post-processing
        with self.analysis_semaphore:
            result = self.post_process(work_dir)

        logger.debug(f"Smilei Simulation finished, got result: {result}, parameters: {', '.join([str(x) for x in par_vec])}")

        # write processed result to current gen file
        with open(
            f"gen{self.generation:0>3d}.csv" if self.generation is not None else "geninit.csv",
            'a'
        ) as gen_file:
            xs = ','.join([str(x) for x in par_vec])
            print(xs, -result, sep=',', file=gen_file)

        logger.debug("Written results to generation file")

        # tidy up - log failure but do nothing about it now - failures probably due to latency on HPC ephemeral store
        shutil.rmtree(
            work_dir,
            onerror=lambda _f, p, e: logger.warning(f"Failed to delete {p} as {e}")
        )

        logger.debug("Attempted to remove temp dir")

        # finally, return result
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['mpi_spawn_lock']
        del state['analysis_semaphore']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.mpi_spawn_lock = threading.Lock()
        self.analysis_semaphore = threading.Semaphore(self.analysis_concurrency)
