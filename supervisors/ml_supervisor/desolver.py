import logging
import numpy as np
import pickle

from scipy.optimize._differentialevolution import DifferentialEvolutionSolver, _MACHEPS
from scipy._lib._util import MapWrapper
from concurrent.futures import ThreadPoolExecutor

from utils import pp_array

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

        workers = ThreadPoolExecutor(max_workers=threads).map

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

        logger.info(f"Initial population complete")
        logger.info(pp_array(self.x))
        logger.info(f"Energy is {-self.population_energies[0]}, convergence is: {self.tol / (self.convergence + _MACHEPS)}")

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

            logger.info("=============================================")
            logger.info(f"            Generation {i} complete")
            logger.info(pp_array(self.x))
            logger.info(f"Energy is {-self.population_energies[0]:.3e}, convergence is: {self.tol / (self.convergence + _MACHEPS)}")
            logger.info("=============================================")
        else:
            iter_exhausted = True

        logger.info("=============================================")
        logger.info("             Optimisation Result")
        logger.info([pp_array(self.x)])
        logger.info(f"{-self.population_energies[0]:.3e}")
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
        self._mapwrapper = MapWrapper(ThreadPoolExecutor(max_workers=self.threads).map)
        self.func = self.smilei_wrapper.run_sim
