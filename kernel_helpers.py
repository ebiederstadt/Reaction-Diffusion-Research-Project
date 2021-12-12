from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.lib.function_base import diff
from scipy.fft import dct
from scipy.integrate import simpson

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
            * np.exp(-(np.square((x - self.distance) / self.width) / 2))
        )

    def compute_at_given_distance(self, distance: float) -> float:
        return (
            self.amplitude
            / np.sqrt(2 * np.pi)
            * np.exp(-(np.square((distance - self.distance) / self.width) / 2))
        )

    def update(self, amplitude: float, distance: float, width: float):
        self.amplitude = amplitude
        self.distance = distance
        self.width = width

        self.kernel = self.compute()

    def diff(self, amplitude, width, distance):
        """Check to see if anything has changed"""
        return not (
            self.amplitude == amplitude
            and self.width == width
            and self.distance == distance
        )


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

        print(f"2D Integral of Kernel: {self.compute_2d_integral()}")

    def partial_update_activator(self, amplitude: float, distance: float, width: float):
        self.activator.update(amplitude, distance, width)
        self.kernel = self.activator.kernel + self.inhibitor.kernel

    def update_inhibitor(self, amplitude: float, distance: float, width: float):
        self.inhibitor.update(amplitude, distance, width)
        self.kernel = self.activator.kernel + self.inhibitor.kernel
        self.update_cache()
        self.integrate()
        self.fourier = self.compute_fourier()

        print(f"2D Integral of Kernel: {self.compute_2d_integral()}")

    def parital_update_inhibitor(self, amplitude: float, distance: float, width: float):
        self.inhibitor.update(amplitude, distance, width)
        self.kernel = self.activator.kernel + self.inhibitor.kernel

    def compute_2d_integral(self):
        """The 2D integral of the kernel is just the sum of the 2D kernel"""

        return self.cache.sum()


@njit
def compute_stimulation(kernel_cache: np.ndarray, kt_matrix: np.ndarray):
    """This method is JIT compiled to machine code, and runs much faster as a result"""

    stimulation_matrix = np.zeros(c.MATRIX_SIZE ** 2)
    kernel_width = c.KERNEL_SIZE * 2
    indexMat = 0
    for j in range(c.MATRIX_SIZE):
        yy = j + c.MATRIX_SIZE - c.KERNEL_SIZE
        for i in range(c.MATRIX_SIZE):
            xx = i + c.MATRIX_SIZE - c.KERNEL_SIZE
            matValue = kt_matrix[indexMat]
            indexMat = indexMat + 1
            indexK = 0
            for q in range(kernel_width):
                y = (yy + q) % c.MATRIX_SIZE
                for p in range(kernel_width):
                    x = (xx + p) % c.MATRIX_SIZE
                    index = c.MATRIX_SIZE * y + x
                    stimulation_matrix[index] += kernel_cache[indexK] * matValue
                    indexK = indexK + 1

    return stimulation_matrix


class ConstantKernel(Kernel):
    def __init__(self, kernel: np.ndarray):
        self.kernel = kernel
        self.integral = self.integrate()
        self.cache = np.zeros((c.KERNEL_SIZE * 2 + 1) * (c.KERNEL_SIZE * 2 + 1))
        self.update_cache()
        self.fourier = self.compute_fourier()

    def update_cache(self):
        self.cache = np.full(
            (c.KERNEL_SIZE * 2 + 1) * (c.KERNEL_SIZE * 2 + 1), self.kernel[0]
        )
