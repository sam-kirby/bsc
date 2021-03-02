import argparse
import pathlib
import threading
import sys

from mpi4py import MPI

import utils

from desolver import DESolver
from goal_functions import max_energy_negated
from smilei_wrapper import SmileiWrapper

# setup logging - split logs into two files
#  - main.log: Generation change and results only
#  - debug.log: Every smilei sim start + finish + everything in debug.log. Noisy.
logger = utils.init_logger()

# Parse arguments
parser = argparse.ArgumentParser(description="Optimise the parameters of a PIC simulation using Differential Evolution")
parser.add_argument(
    "-b",
    "--bound",
    action="append",
    nargs=2,
    type=float,
    required=True,
    help="You must specify at least one bound. Specifying only one bound assumes all dims have the same bound",
    dest="bounds"
)
parser.add_argument(
    "-d",
    "--dims",
    type=int,
    required=True,
    help = "The dimensionality of the optimisation problem"
)
parser.add_argument(
    "-i",
    "--maxiter",
    default=1000,
    type=int,
    help="The maximum number of iterations"
)
parser.add_argument(
    "-p",
    "--popsize",
    default=15,
    type=int,
    help="A population multiplier, len(bounds) * popsize = number of simulations per generation"
)
parser.add_argument(
    "--maxenergy",
    action="store_const",
    const=max_energy_negated,
    dest="goal_func"
)
parser.add_argument(
    "--usize",
    type=int,
    help="The size of the MPI universe if it cannot be determined automatically"
)
parser.add_argument(
    "--athreads",
    default=4,
    type=int,
    help="The number of threads that can perform analysis at a time"
)
parser.add_argument(
    "namelist",
    type=pathlib.Path,
    help="A path to a valid Smilei namelist"
)

args = parser.parse_args()

# parse bounds
if (supplied_bounds := len(args.bounds)) == 1:
    bounds = [(l, u) for [l, u] in args.bounds * args.dims]
elif supplied_bounds == args.dims:
    bounds = [(l, u) for [l, u] in args.bounds]
else:
    logger.error("Invalid bounds - must specify either a single bound or N bounds")
    sys.exit(1)

# check a goal function has been set
if (goal_func := args.goal_func) is None:
    logger.warning("No goal function was set, using default")
    goal_func = max_energy_negated

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

# construct SmileiWrapper
smilei_wrapper = SmileiWrapper(args.namelist, goal_func, args.athreads)

# construct Solver
solver = DESolver(
    smilei_wrapper,
    bounds,
    threads=usize - 1,
    maxiter=args.maxiter,
    popsize=args.popsize
)

solver.prepare()

solver.optimise()
