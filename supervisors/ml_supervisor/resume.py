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
    "--usize",
    type=int,
    help="The size of the MPI universe if it cannot be determined automatically"
)
parser.add_argument(
    "pickled_solver",
    type=pathlib.Path,
    help="Path to the pickled solver to resume from"
)

args = parser.parse_args()

# MPI Setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if (usize := comm.Get_attr(MPI.UNIVERSE_SIZE)) is None:
    logger.warning("Unable to determine universe size automatically, make sure this has been set correctly")
    usize = args.usize

if rank == 0:
    logger.info(f"Supervisor is running at rank {rank}. The universe contains {usize} nodes.")
else:
    logger.error(f"Supervisor expected to be running at rank 0, but is actually rank {rank}")
    sys.exit(1)

comm.Set_errhandler(MPI.ERRORS_ARE_FATAL)
MPI.COMM_SELF.Set_errhandler(MPI.ERRORS_ARE_FATAL)

with open(args.pickled_solver, 'rb') as pickle_file:
    solver = pickle.load(pickle_file)

solver.optimise()
