import matplotlib.pyplot as plt
import numpy as np

from collections import deque
from glob import glob
from itertools import islice
from matplotlib.animation import FuncAnimation
from pathlib import Path
from scipy.constants import c, e, epsilon_0, m_e, pi

from stubs import load_stub

wavelength = 800.e-9
omega = 2 * pi * c / wavelength

filepath = Path("density_ml_best1bin_150")
solver_files = sorted(glob(f"{filepath.absolute()}/solver*.pickle"))  # file paths aren't sorted on ext4
solver_files = np.roll(solver_files, 1)  # move init to start of list

solvers = []

for sf in solver_files:
    with open(sf, "rb") as sh:
        solvers.append(load_stub(sh))

populations = []
energies = []

for solver in solvers:
    populations.append(solver._scale_parameters(solver.population))
    energies.append(-solver.population_energies)

populations = np.array(populations)
energies = np.array(energies)

print(populations[-1][0], energies[-1][0])

# Energy Evolution
results = []

for energy in energies:
    results.append((np.min(energy), (np.average(energy), np.std(energy)), np.max(energy)))

fig, ax = plt.subplots(num=1)
minimums, averages, maximums = zip(*results)
avgs, errs = zip(*averages)
avgs = np.array(avgs)
errs = np.array(errs)
ax.plot(minimums, label="Minimum", color="blue")
ax.plot(avgs, label="Average", color="black")
ax.fill_between(np.arange(len(avgs)), avgs - errs, avgs + errs, facecolor="gray", alpha=0.4)
ax.plot(maximums, label="Maximum", color="red")
ax.set_xlim((0, len(energies)))
ax.set_ylim((0, np.max(maximums) * 1.1))

ax.set_xlabel("Generation")
ax.set_ylabel("Energy (arb. units)")
ax.legend()

fig.savefig("energy_evolution.png", dpi=400)

# Pop Evolution
fig, ax_anim = plt.subplots(num=2)
x_axis = (np.arange(len(populations[0][0])) + 0.5) * 5e-7
generation = 0

def update(pop):
    global generation
    ax_anim.clear()
    ax_anim.set_title(f"Generation {generation}")
    generation += 1
    ax_anim.set_xlim((0, 5e-6))
    ax_anim.set_ylim((0, 10))
    ax_anim.set_xlabel(r"x ($\mu m$)")
    ax_anim.set_ylabel(r"Density (relative to $n_{crit}$)")
    for profile in pop[1:]:
        ax_anim.bar(x_axis, profile, width=5e-7, color="None", edgecolor="black", alpha=0.2)
    
    ax_anim.plot((0, 5e-6), (1, 1), color="cyan", alpha=0.5)
    ax_anim.bar(x_axis, pop[0], width=5e-7, color="None", edgecolor="red", linewidth=2, label="Best")

    ax_anim.legend(loc="upper left")

ani = FuncAnimation(fig, update, frames=populations, interval=50).save("pop_evolution.gif", dpi=400)

# Trial Evolution
gen_files = sorted(glob(f"{filepath.absolute()}/gen*.csv"))
gen_files = np.roll(gen_files, 1)

trials = []

for gf in gen_files:
    with open(gf, 'r') as gh:
        trials.append(np.loadtxt(gh, delimiter=",")[:,:10])

fig, ax_anim = plt.subplots(num=3)
generation = 0

ani = FuncAnimation(fig, update, frames=trials, interval=50).save("trial_evolution.gif", dpi=400)

# Best Evolution
fig, ax_anim = plt.subplots(num=4)
bests = deque([], 50)
generation = 0

def update(pop):
    global generation
    ax_anim.clear()
    ax_anim.set_title(f"Best at generation {generation}")
    generation += 1
    ax_anim.set_xlim((0, 5e-6))
    ax_anim.set_ylim((0, 10))
    ax_anim.set_xlabel(r"x ($\mu m$)")
    ax_anim.set_ylabel(r"Density (relative to $n_{crit}$)")
    bests.appendleft(pop[0])
    for i, profile in islice(enumerate(bests), 1, len(bests)):
        ax_anim.bar(x_axis, profile, width=5e-7, color="None", edgecolor="black", alpha=1/(i + 1))
    ax_anim.bar(x_axis, bests[0], width=5e-7, color="None", edgecolor="red")

ani = FuncAnimation(fig, update, frames=populations, interval=200).save("best_evolution.gif", dpi=400)
