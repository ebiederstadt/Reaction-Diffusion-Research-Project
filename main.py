import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.managers import SharedMemoryManager
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from shared_ndarray2 import SharedNDArray

import constants as c
from image_processing import save_fig, write_figures
from kernel_helpers import KernelPortion, set_kernel_cache, integrate_kernel


x = np.linspace(0, c.KERNEL_SIZE)

# Starting values for the kernel and the simulation matrix
kt_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)
activator = KernelPortion(21, 0, 2.06)
inhibitor = KernelPortion(-6.5, 4.25, 1.38)

kernel = activator.kernel + inhibitor.kernel
kernel_cache = np.zeros((c.KERNEL_SIZE * 2 + 1) * (c.KERNEL_SIZE * 2 + 1))
set_kernel_cache(kernel_cache, activator, inhibitor)


def save_figures(event):
    """Save all the figures"""
    time_stamp = datetime.timestamp(datetime.now())
    write_figures(kt_matrix, kernel, activator, inhibitor)
    save_fig(f"{time_stamp}_fourier_transform.png", fig, ax[1][0])


def matrixIndex(x, y):
    return c.MATRIX_SIZE * y + x


def simulate(event):
    global kt_matrix

    with SharedMemoryManager() as mem_mgr:
        print("Starting simulation...")
        start = perf_counter()
        stimulation_matrix = SharedNDArray.from_array(
            mem_mgr, np.zeros(c.MATRIX_SIZE ** 2)
        )
        with mp.Pool() as pool:
            pool.starmap(
                compute_cell_stimulation,
                [
                    (j, i, stimulation_matrix)
                    for j in range(c.MATRIX_SIZE)
                    for i in range(c.MATRIX_SIZE)
                ],
            )

    np.clip(stimulation_matrix.get(), c.MIN_STIMULATION, c.MAX_STIMULATION)
    kt_matrix = kt_matrix * c.DECAY_RATE + stimulation_matrix.get()
    end = perf_counter()
    print("Simulation finished")
    print(f"Simulation took {end - start} seconds")

    ax[0][0].imshow(np.reshape(kt_matrix, (200, 200)), interpolation="none")


def compute_cell_stimulation(j, i, stimulation_matrix: SharedNDArray) -> float:
    """Compute the stimulation received by a single cell"""

    kernel_width = c.KERNEL_SIZE * 2
    yy = j + c.MATRIX_SIZE - c.KERNEL_SIZE
    xx = i + c.MATRIX_SIZE - c.KERNEL_SIZE
    mat_value = kt_matrix[j * c.MATRIX_SIZE + i]

    for q in range(kernel_width):
        y = (yy + q) % c.MATRIX_SIZE
        for p in range(kernel_width):
            x = (xx + p) % c.MATRIX_SIZE
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
    ax[0][1].set_xlim(0, c.KERNEL_SIZE)
    ax[0][1].grid(True)
    ax[0][1].plot(x, activator.kernel, label="Activator", linestyle="dashed")
    ax[0][1].plot(x, inhibitor.kernel, label="Inhibitor", linestyle="dashed")
    ax[0][1].plot(x, kernel, label="Kernel")
    ax[0][1].legend()

    ax[1][0].set_title("Fourier Transform of the Kernel")
    ax[1][0].grid(True)

    ax[1][1].axis("off")

    # Show the Integral of the Kernel
    integral = integrate_kernel(kernel)
    plt.text(0, 0.2, f"{integral:.3f}")

    # Interactive widgets and buttons
    ax_save = plt.axes([0.5, 0.4, 0.1, 0.075])
    button_save = Button(ax_save, "Save Figures")
    button_save.on_clicked(save_figures)

    ax_compute = plt.axes([0.6, 0.4, 0.1, 0.075])
    button_compute = Button(ax_compute, "Start Calculation")
    button_compute.on_clicked(simulate)

    ax_repeat = plt.axes([0.7, 0.4, 0.1, 0.075])
    button_repeat = Button(ax_repeat, "Calculate x10")
    button_repeat.on_clicked(lambda e: [simulate(e) for _ in range(10)])

    plt.show()
