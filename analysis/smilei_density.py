import argparse
import happi
import matplotlib.pyplot as plt
import pathlib

from copy import copy

parser = argparse.ArgumentParser(description="Optimise the parameters of a PIC simulation using Differential Evolution")
parser.add_argument(
    "rdir",
    type=pathlib.Path,
    help="A path to a directory containing Smilei results"
)
args = parser.parse_args()

data = happi.Open(args.rdir.absolute())


