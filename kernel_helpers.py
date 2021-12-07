from dataclasses import dataclass
from scipy.integrate import simpson
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


def kernel_index(x, y):
    return (c.KERNEL_SIZE * 2 + 1) * y + x


def set_kernel_cache(
    kernel_cache: np.ndarray, activator: KernelPortion, inhibitor: KernelPortion
):
    for p in range(-c.KERNEL_SIZE, c.KERNEL_SIZE + 1):
        for q in range(-c.KERNEL_SIZE, c.KERNEL_SIZE + 1):
            distance = np.sqrt(p ** 2 + q ** 2)
            kernel_cache[
                kernel_index(p + c.KERNEL_SIZE, q + c.KERNEL_SIZE)
            ] = activator.compute_at_given_distance(
                distance
            ) + inhibitor.compute_at_given_distance(
                distance
            )


def integrate_kernel(kernel):
    x = np.linspace(0, c.KERNEL_SIZE)
    return simpson(kernel, x)
