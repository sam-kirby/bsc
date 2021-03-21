import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from scipy.constants import pi
from scipy.fft import fftshift, ifft
from typing import List, Optional

@dataclass
class ChirpedLaser(object):
    omega_0: float
    tau: float
    a_0: float
    beta: Optional[List[float]] = None
    dt: Optional[float] = None
    E_t: Optional[np.ndarray] = None
    t_0: Optional[int] = None
    t_end: Optional[int] = None

    def __post_init__(self):
        if self.beta is not None:
            self.initialise(*self.beta)

    def initialise(self, *beta: List[float], timestep_si=None):
        self.beta = beta

        omega_max = 2**10 * self.omega_0
        omega = np.linspace(0, omega_max, 2**24, dtype=np.complex128)
        omega_prime = omega - self.omega_0
        exponent = -1./4. * omega_prime**2 * self.tau**2

        E_omega_g = self.tau / np.sqrt(2) * np.exp(exponent)
        E_t_g = fftshift(ifft(E_omega_g))
        scale = self.a_0 / np.max(np.real(E_t_g))

        for n, beta in enumerate(beta):
            if beta == 0.:
                continue
            exponent -= 1.j * beta / np.math.factorial(n) * omega_prime**n
        E_omega = self.tau / np.sqrt(2) * np.exp(exponent)

        E_t = scale * np.real(fftshift(ifft(E_omega)))

        self.dt = 2 * pi / omega_max

        t_0 = len(E_t) - next(i for i, E in enumerate(E_t[::-1]) if np.abs(E) > 1e-2) - 1
        t_end = next(i for i, E in enumerate(E_t) if np.abs(E) > 1e-2)

        self.E_t = E_t[t_0 : t_end : -1]
    
    def at_sim_time(self, t: float) -> float:
        index = int(t / self.omega_0 / self.dt)
        try:
            return self.E_t[index]
        except IndexError:
            return 0.

    def get_peak_offset(self) -> float:
        return np.argmax(self.E_t) * self.dt

    def plot(self, with_carrier: bool, with_envelope: bool):
        time = (np.arange(len(self.E_t)) - len(self.E_t) / 2) * self.dt
        plt.plot(time, self.E_t)
        if with_carrier:
            plt.plot(time, self.a_0 * np.cos(self.omega_0 * time), alpha=0.3)
        if with_envelope:
            plt.plot(time, self.a_0 * np.exp(-time**2/self.tau**2))
        plt.show()
