from dataclasses import dataclass
from scipy.integrate import simpson
from scipy.fft import dct
import numpy as np

import constants as c


@dataclass
class KernelPortion:
    amplitude: float
    distance: float
    width: float

    def __init__(self, amplitude: float, distance: float, width: float):
        self.amplitude = amplitude
        self.distance = distance
        self.width = width

        self.kernel = self.compute()

    def compute(self):
        """Compute the value of the gaussian function"""

        x = np.linspace(0, c.KERNEL_SIZE)
        return (
            self.amplitude
            / np.sqrt(2 * np.pi)
            * np.exp(-(((x - self.distance) ** 2 / self.width) / 2))
        )

    def compute_at_given_distance(self, distance: float) -> float:
        return (
            self.amplitude
            / np.sqrt(2 * np.pi)
            * np.exp(-(((distance - self.distance) ** 2 / self.width) / 2))
        )

    def update(self, amplitude: float, distance: float, width: float):
        self.amplitude = amplitude
        self.distance = distance
        self.width = width

        self.kernel = self.compute()


class Kernel:
    def __init__(self):
        self.activator = KernelPortion(21, 0, 2.06)
        self.inhibitor = KernelPortion(-6.5, 4.25, 1.38)
        self.kernel = self.activator.kernel + self.inhibitor.kernel
        self.integral = self.integrate()
        self.cache = np.zeros((c.KERNEL_SIZE * 2 + 1) * (c.KERNEL_SIZE * 2 + 1))
        self.update_cache()
        self.fourier = self.compute_fourier()

    def _index(self, x, y):
        return (c.KERNEL_SIZE * 2 + 1) * y + x

    def integrate(self):
        x = np.linspace(0, c.KERNEL_SIZE)
        return simpson(self.kernel, x)

    def compute_fourier(self):
        yf = dct(self.kernel)
        return yf

    def update_cache(self):
        for p in range(-c.KERNEL_SIZE, c.KERNEL_SIZE + 1):
            for q in range(-c.KERNEL_SIZE, c.KERNEL_SIZE + 1):
                distance = np.sqrt(p ** 2 + q ** 2)
                self.cache[
                    self._index(p + c.KERNEL_SIZE, q + c.KERNEL_SIZE)
                ] = self.activator.compute_at_given_distance(
                    distance
                ) + self.inhibitor.compute_at_given_distance(
                    distance
                )

    def update_activator(self, amplitude: float, distance: float, width: float):
        self.activator.update(amplitude, distance, width)
        self.kernel = self.activator.kernel + self.inhibitor.kernel
        self.update_cache()
        self.integrate()
        self.fourier = self.compute_fourier()

    def update_inhibitor(self, amplitude: float, distance: float, width: float):
        self.inhibitor.update(amplitude, distance, width)
        self.kernel = self.activator.kernel + self.inhibitor.kernel
        self.update_cache()
        self.integrate()
        self.fourier = self.compute_fourier()
