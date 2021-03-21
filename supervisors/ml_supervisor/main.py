import argparse
import pathlib
import threading
import sys

from mpi4py import MPI

import utils

from desolver import DESolver
from goal_functions import max_energy_negated, screen_dep_energy_negated, load_result_from_file
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
    "--best1bin",
    action="store_const",
    const="best1bin",
    dest="strategy",
    help="Sets the algorithm strategy to best1bin - population members are produced by mutating the best of the previous generation using 1 difference vectors with binomial crossover"
)
parser.add_argument(
    "--best2bin",
    action="store_const",
    const="best2bin",
    dest="strategy",
    help="Sets the algorithm strategy to best2bin - population members are produced by mutating the best of the previous generation using 2 difference vectors with binomial crossover"
)
parser.add_argument(
    "--rand1bin",
    action="store_const",
    const="rand1bin",
    dest="strategy",
    help="Sets the algorithm strategy to rand1bin - population members are produced by mutating a random member of the previous generation using 1 difference vectors with binomial crossover"
)
parser.add_argument(
    "--currenttobest1bin",
    action="store_const",
    const="currenttobest1bin",
    dest="strategy",
    help="Sets the algorithm strategy to currenttobest1bin - population members are produced by mutating the current member of the previous generation using 2 difference vectors (one being the difference between the current and the best) with binomial crossover"
)
parser.add_argument(
    "--mutation",
    default=1.,  # Note that SciPy's default is (0.5, 1.)
    type=float,
    help="The mutation constant or differential weight"
)
parser.add_argument(
    "--dither",
    type=float,
    default=None,
    help="If set, enables dithering between the value of the mutation parameter and this value"
)
parser.add_argument(
    "--crossover",
    type=float,
    default=0.7,
    help="The crossover probability, increasing this value increases the number of mutants that progress into the next generation"
)
parser.add_argument(
    "--maxenergy",
    action="store_const",
    const=max_energy_negated,
    dest="goal_func"
)
parser.add_argument(
    "--depenergy",
    action="store_const",
    const=screen_dep_energy_negated,
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
    "--maxsims",
    type=int,
    help="The maximum number of simulations to run before terminating. The optimisation is resumable if this limit is exhausted; a further `maxsims` simulations will be run"
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

# parse strategy
if (strategy := args.strategy) is None:
    strategy = "best1bin"
    logger.info(f"No strategy specified, using '{strategy}'")

# parse mutation
if args.dither is None:
    mutation = args.mutation
else:
    mutation = (args.mutation, args.dither)

# check a goal function has been set
if (goal_func := args.goal_func) is None:
    logger.warning("No goal function was set, assuming analysis will be performed by Smilei")
    goal_func = load_result_from_file

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
    popsize=args.popsize,
    strategy=strategy,
    mutation=mutation,
    recombination=args.crossover,
    max_sims=args.maxsims
)

solver.prepare()

solver.optimise()
