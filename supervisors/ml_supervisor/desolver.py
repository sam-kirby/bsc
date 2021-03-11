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
        strategy,
        maxiter,
        popsize,
        mutation,
        recombination,
        max_sims,
        tol=0.01,
        atol=0,
        init='latinhypercube',
        seed=None,
        constraints=()
    ):
        self.smilei_wrapper = smilei_wrapper
        self.threads = threads
        self.max_sims = max_sims

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
        logger.info(f"Energy is {-self.population_energies[0]:.3e}, convergence is: {self.tol / (self.convergence + _MACHEPS):.3e}")

    def optimise(self):
        logger = logging.getLogger("supervisor")
        
        # Optimise
        logger.info("Beginning optimisation")

        gens_exhausted = False
        sims_exhausted = False
        (start_gen, sims_run) = (0, self.num_population_members) if (gen := self.smilei_wrapper.generation) is None else (gen + 1, 0)
        for i in range(start_gen, self.maxiter):
            self.smilei_wrapper.generation = i

            if self.max_sims is None or (sims_run := sims_run + self.num_population_members) < self.max_sims:
                next(self)
            else:
                sims_exhausted = True
                break

            # checkpoint
            with open(f"solver{i:0>3d}.pickle", "wb") as pickle_file:
                pickle.dump(self, pickle_file)

            if self.converged():
                break

            logger.info("=============================================")
            logger.info(f"            Generation {i} complete")
            logger.info(pp_array(self.x))
            logger.info(f"Energy is {-self.population_energies[0]:.3e}, convergence is: {self.tol / (self.convergence + _MACHEPS):.3e}")
            logger.info("=============================================")
        else:
            gens_exhausted = True

        logger.info("=============================================")
        logger.info("             Optimisation Result")
        logger.info([pp_array(self.x)])
        logger.info(f"{-self.population_energies[0]:.3e}")
        logger.info("=============================================")

        if gens_exhausted:
            logger.info(f"Generation limit exhausted after {i} generations")
        elif sims_exhausted:
            logger.info(f"Simulation limit exhausted after {i} generations")
        else:
            logger.info(f"Optimisation converged after {i} generations!")


    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_mapwrapper']
        del state['func']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mapwrapper = MapWrapper(ThreadPoolExecutor(max_workers=self.threads).map)
        self.func = self.smilei_wrapper.run_sim
