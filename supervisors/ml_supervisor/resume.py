import argparse
import os
import pathlib
import pickle

import utils

# setup logging - split logs into two files
#  - main.log: Generation change and results only
#  - debug.log: Every smilei sim start + finish + everything in debug.log. Noisy.
logger = utils.init_logger()

parser = argparse.ArgumentParser(description="Optimise the parameters of a PIC simulation using Differential Evolution")
parser.add_argument(
    "pickled_solver",
    type=pathlib.Path,
    help="Path to the pickled solver to resume from"
)

args = parser.parse_args()

with open(args.pickled_solver, 'rb') as pickle_file:
    solver = pickle.load(pickle_file)

solver.optimise()
