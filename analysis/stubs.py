import pickle

from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

__all__ = ["load_stub"]

class DESolver(DifferentialEvolutionSolver):
    pass

class SmileiWrapper:
    pass

def load_result_from_file():
    pass

class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "desolver" or module == "smilei_wrapper" or module == "goal_functions":
            module = __name__
        return super().find_class(module, name)

def load_stub(f):
    return StubUnpickler(f).load()
    