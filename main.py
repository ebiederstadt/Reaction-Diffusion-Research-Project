import multiprocessing as mp
from datetime import datetime
from multiprocessing.managers import SharedMemoryManager
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

import constants as c
from image_processing import write_figures
from kernel_helpers import Kernel, compute_stimulation

x = np.linspace(0, c.KERNEL_SIZE)

# Starting values for the kernel and the simulation matrix
kt_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)
kernel = Kernel()

fig, ax = plt.subplots(2, 2)
# Show the Integral of the Kernel
integral_text = fig.text(
    0.5, 0.1, f"Integrated Value of the Kernel: {kernel.integral:.3f}"
)


def save_figures(event):
    """Save all the figures"""
    time_stamp = datetime.timestamp(datetime.now())
    write_figures(kt_matrix, kernel)


def simulate(event):
    global kt_matrix

    print("Starting simulation")
    start = perf_counter()
    stimulation_matrix = compute_stimulation(kernel.cache, kt_matrix)

    np.clip(
        stimulation_matrix, c.MIN_STIMULATION, c.MAX_STIMULATION, out=stimulation_matrix
    )
    print(
        f"stimulation max: {stimulation_matrix.max()} min: {stimulation_matrix.min()}, mean: {stimulation_matrix.mean()}"
    )
    kt_matrix = kt_matrix * c.DECAY_RATE + stimulation_matrix / 100
    end = perf_counter()
    print("Simulation finished")
    print(f"Simulation took {end - start} seconds")

    global fig, ax
    ax = ax.ravel()
    ax[0].imshow(
        np.reshape(kt_matrix, (c.MATRIX_SIZE, c.MATRIX_SIZE)), interpolation="none"
    )
    plt.draw()


def update_activator_from_textbox(text):
    try:
        amplitude, width, distance = [float(x) for x in text.split(",")]
    except ValueError:
        print("Invalid input given to activator textbox")
        print(text)
        return

    kernel.update_activator(amplitude, distance, width)

    # Redraw the figure
    global fig, ax
    ax = ax.ravel()
    ax[1].lines[0].set_ydata(kernel.kernel)
    ax[1].lines[1].set_ydata(kernel.activator.kernel)

    ax[2].lines[0].set_ydata(kernel.fourier)

    integral_text.set_text(f"Integrated Value of the Kernel: {kernel.integral:.3f}")

    plt.draw()


def update_inhibitor_from_textbox(text):
    try:
        amplitude, width, distance = [float(x) for x in text.split(",")]
    except ValueError:
        print("Invalid input given to inhibitor textbox")
        print(text)
        return

    kernel.update_inhibitor(amplitude, distance, width)

    # Redraw the figure
    global fig, ax
    ax = ax.ravel()
    ax[1].lines[0].set_ydata(kernel.kernel)
    ax[1].lines[2].set_ydata(kernel.inhibitor.kernel)

    ax[2].lines[0].set_ydata(kernel.fourier)

    integral_text.set_text(f"Integrated Value of the Kernel: {kernel.integral:.3f}")

    plt.draw()


def randomize(event):
    global kt_matrix
    global fig, ax

    kt_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)

    ax = ax.ravel()
    ax[0].imshow(
        np.reshape(kt_matrix, (c.MATRIX_SIZE, c.MATRIX_SIZE)), interpolation="none"
    )
    plt.draw()


if __name__ == "__main__":
    ax[0][0].set_title("Reaction Diffusion Result")
    ax[0][0].imshow(
        np.reshape(kt_matrix, (c.MATRIX_SIZE, c.MATRIX_SIZE)), interpolation="none"
    )

    ax[0][1].set_title("Kernel (Activator + Inhibitor)")
    ax[0][1].set_xlim(0, c.KERNEL_SIZE)
    ax[0][1].grid(True)
    ax[0][1].plot(x, kernel.kernel, label="Kernel")
    ax[0][1].plot(x, kernel.activator.kernel, label="Activator", linestyle="dashed")
    ax[0][1].plot(x, kernel.inhibitor.kernel, label="Inhibitor", linestyle="dashed")
    ax[0][1].legend()

    ax[1][0].set_title("Fourier Transform of the Kernel")
    ax[1][0].grid(True)
    ax[1][0].plot(kernel.fourier)
    ax[1][0].set_xlim(0, 20)

    ax[1][1].axis("off")

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

    ax_randomize = plt.axes([0.8, 0.4, 0.1, 0.075])
    button_randomize = Button(ax_randomize, "Randomize")
    button_randomize.on_clicked(randomize)

    ax_activator_input = plt.axes([0.65, 0.3, 0.1, 0.075])
    activator_params = TextBox(ax_activator_input, "A(x) (Amplitude, Width, Distance):")
    activator_params.on_submit(update_activator_from_textbox)

    ax_inhibitor_input = plt.axes([0.65, 0.2, 0.1, 0.075])
    inhibitor_params = TextBox(ax_inhibitor_input, "I(x) (Amplitude, Width, Distance):")
    inhibitor_params.on_submit(update_inhibitor_from_textbox)

    plt.show()
