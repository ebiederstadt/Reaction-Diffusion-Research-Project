import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.managers import SharedMemoryManager
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button
from shared_ndarray2 import SharedNDArray

from image_processing import save_fig

# Constants
MATRIX_SIZE = 200
KERNEL_SIZE = 20

MIN_STIMULATION = 0
MAX_STIMULATION = 5
DECAY_RATE = 0.9


@dataclass
class KernelPortion:
    amplitude: float
    distance: float
    width: float

    def __init__(self, amplitude, distance, width):
        self.amplitude = amplitude
        self.distance = distance
        self.width = width

        self.kernel = self.update_kernel_portion()

    def update_kernel_portion(self):
        """Returns one half of the kernel (either the activator or the inhibitor)"""

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


def kernel_index(x, y):
    return (KERNEL_SIZE * 2 + 1) * y + x


def set_kernel_cache():
    for p in range(-KERNEL_SIZE, KERNEL_SIZE + 1):
        for q in range(-KERNEL_SIZE, KERNEL_SIZE + 1):
            distance = np.sqrt(p ** 2 + q ** 2)
            kernel_cache[
                kernel_index(p + KERNEL_SIZE, q + KERNEL_SIZE)
            ] = activator.compute_at_given_distance(
                distance
            ) + inhibitor.compute_at_given_distance(
                distance
            )


x = np.linspace(0, KERNEL_SIZE)

# Starting values for the kernel and the simulation matrix
kt_matrix = np.random.rand(MATRIX_SIZE * MATRIX_SIZE)
activator = KernelPortion(21, 0, 2.06)
inhibitor = KernelPortion(-6.5, 4.25, 1.38)

kernel = activator.kernel + inhibitor.kernel
kernel_cache = np.zeros((KERNEL_SIZE * 2 + 1) * (KERNEL_SIZE * 2 + 1))
set_kernel_cache()


def save_figures(event):
    """Save all the figures"""
    time_stamp = datetime.timestamp(datetime.now())
    save_fig(f"{time_stamp}_reaction_diffusion_result.png", fig, ax[0][0])
    save_fig(f"{time_stamp}_kernel.png", fig, ax[0][1])
    save_fig(f"{time_stamp}_fourier_transform.png", fig, ax[1][0])


def matrixIndex(x, y):
    return MATRIX_SIZE * y + x


def simulate(event):
    global kt_matrix

    with SharedMemoryManager() as mem_mgr:
        print("Starting simulation...")
        start = perf_counter()
        stimulation_matrix = SharedNDArray.from_array(
            mem_mgr, np.zeros(MATRIX_SIZE ** 2)
        )
        with mp.Pool() as pool:
            pool.starmap(
                compute_cell_stimulation,
                [
                    (j, i, stimulation_matrix)
                    for j in range(MATRIX_SIZE)
                    for i in range(MATRIX_SIZE)
                ],
            )

    np.clip(stimulation_matrix.get(), MIN_STIMULATION, MAX_STIMULATION)
    kt_matrix = kt_matrix * DECAY_RATE + stimulation_matrix.get()
    end = perf_counter()
    print("Simulation finished")
    print(f"Simulation took {end - start} seconds")

    ax[0][0].imshow(np.reshape(kt_matrix, (200, 200)), interpolation="none")


def compute_cell_stimulation(j, i, stimulation_matrix: SharedNDArray) -> float:
    """Compute the stimulation received by a single cell

    TODO: This does not work at the moment because the stimulation matrix needs to be shared across processes"""

    kernel_width = KERNEL_SIZE * 2
    yy = j + MATRIX_SIZE - KERNEL_SIZE
    xx = i + MATRIX_SIZE - KERNEL_SIZE
    mat_value = kt_matrix[j * MATRIX_SIZE + i]

    for q in range(kernel_width):
        y = (yy + q) % MATRIX_SIZE
        for p in range(kernel_width):
            x = (xx + p) % MATRIX_SIZE
            index = matrixIndex(x, y)
            # This is the core idea: we do the numerical integration at this stage, not at the other stage
            stimulation_matrix[index] = (
                stimulation_matrix[index]
                + kernel_cache[q * kernel_width + p] * mat_value
            )


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2)
    ax[0][0].set_title("Reaction Diffusion Result")
    ax[0][0].imshow(np.reshape(kt_matrix, (200, 200)), interpolation="none")

    ax[0][1].set_title("Kernel (Activator + Inhibitor)")
    ax[0][1].set_xlim(0, KERNEL_SIZE)
    ax[0][1].grid(True)
    ax[0][1].plot(x, activator.kernel, label="Activator", linestyle="dashed")
    ax[0][1].plot(x, inhibitor.kernel, label="Inhibitor", linestyle="dashed")
    ax[0][1].plot(x, kernel, label="Kernel")
    ax[0][1].legend()

    ax[1][0].set_title("Fourier Transform of the Kernel")
    ax[1][0].grid(True)

    ax[1][1].axis("off")

    # Interactive widgets and buttons
    ax_save = plt.axes([0.5, 0.4, 0.1, 0.075])
    button_save = Button(ax_save, "Save Figures")
    button_save.on_clicked(save_figures)

    ax_compute = plt.axes([0.6, 0.4, 0.1, 0.075])
    button_compute = Button(ax_compute, "Start Calculation")
    button_compute.on_clicked(simulate)

    plt.show()
