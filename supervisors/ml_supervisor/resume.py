import argparse
import os
import pathlib
import pickle
import sys

from mpi4py import MPI

import utils

# setup logging - split logs into two files
#  - main.log: Generation change and results only
#  - debug.log: Every smilei sim start + finish + everything in debug.log. Noisy.
logger = utils.init_logger()

parser = argparse.ArgumentParser(description="Optimise the parameters of a PIC simulation using Differential Evolution")
parser.add_argument(
    "--solver",
    type=pathlib.Path,
    help="Path to the pickled solver to resume from"
)

args = parser.parse_args()

# MPI Setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if (usize := comm.Get_attr(MPI.UNIVERSE_SIZE)) is None:
    logger.warning("Unable to determine universe size automatically, if this doesn't match the original simulation you'll have problems")

if rank == 0:
    logger.info(f"Supervisor is running at rank {rank}. The universe contains {usize if usize else 'an unknown number of'} nodes.")
else:
    logger.error(f"Supervisor expected to be running at rank 0, but is actually rank {rank}")
    sys.exit(1)

comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)
MPI.COMM_SELF.Set_errhandler(MPI.ERRORS_ARE_FATAL)

if (solver_file := args.solver) is None:
    solver_file = sorted(
        map(
            lambda f: f.name,
            filter(
                lambda f: f.is_file() and "solver" in f.name and "init" not in f.name,
                os.scandir()
            )
        ),
        reverse=True
    )[0]

    start_gen = int(solver_file[6:9]) + 1

    if f"gen{start_gen:0>3d}.csv" in os.listdir():
        logger.error("Expected to resume at start of generation")
        sys.exit(1)

with open(solver_file, 'rb') as pickle_file:
    solver = pickle.load(pickle_file)

solver.optimise()
