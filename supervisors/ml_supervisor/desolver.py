import logging
import numpy as np
import pickle

from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy._lib._util import MapWrapper
from multiprocessing.pool import ThreadPool

class DESolver(DifferentialEvolutionSolver):
    def __init__(self,
        smilei_wrapper,
        bounds,
        threads,
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        init='latinhypercube',
        atol=0,
        constraints=()
    ):
        self.smilei_wrapper = smilei_wrapper
        self.threads = threads

        workers = ThreadPool(processes=threads).map

        super().__init__(
            smilei_wrapper.run_sim,
            bounds,
            args=(),
            strategy=strategy,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            mutation=mutation,
            recombination=recombination,
            seed=seed,
            maxfun=np.inf,
            callback=None,
            disp=False,
            polish=False,
            init=init,
            atol=atol,
            updating='deferred',
            workers=workers,
            constraints=constraints
        )

    def prepare(self):
        logger = logging.getLogger("supervisor")

        logger.info("Preparing initial population")

        if np.all(np.isinf(self.population_energies)):
            self.feasible, self.constraint_violation = (
                self._calculate_population_feasibilities(self.population)
            )

            # only work out population energies for feasible solutions
            self.population_energies[self.feasible] = (
                self._calculate_population_energies(
                    self.population[self.feasible]
                )
            )

            self._promote_lowest_energy()

        # checkpoint
        with open(f"solverinit.pickle", "wb") as pickle_file:
            pickle.dump(self, pickle_file)

        logger.info(f"Initial population complete, current best density profile is:")
        logger.info([str(x) for x in self.x])
        logger.info(f"Energy is {-self.population_energies[0]}, convergence is: {self.convergence}")

    def optimise(self):
        logger = logging.getLogger("supervisor")
        
        # Optimise
        logger.info("Beginning optimisation")

        iter_exhausted = False
        for i in range(0 if (gen := self.smilei_wrapper.generation) is None else gen + 1, self.maxiter):
            self.smilei_wrapper.generation = i

            next(self)

            # checkpoint
            with open(f"solver{i:0>3d}.pickle", "wb") as pickle_file:
                pickle.dump(self, pickle_file)

            if self.converged():
                break

            logger.info(f"Generation {i} complete, current best density profile is:")
            logger.info([str(x) for x in self.x])
            logger.info(f"Energy is {-self.population_energies[0]}, convergence is: {self.convergence}")
        else:
            iter_exhausted = True

        logger.info("=============================================")
        logger.info("             Optimisation Result             ")
        logger.info([str(x) for x in self.x])
        logger.info(-self.population_energies[0])
        logger.info("=============================================")

        if iter_exhausted:
            logger.info("Solver exhausted the iteration limit")
        else:
            logger.info(f"Solver converged after {i} iterations!")


    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_mapwrapper']
        del state['func']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mapwrapper = MapWrapper(ThreadPool(processes=self.threads).map)
        self.func = self.smilei_wrapper.run_sim
