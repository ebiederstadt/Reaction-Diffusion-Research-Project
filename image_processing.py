import os
import csv
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import constants as c


def write_figures(kt_matrix, kernel):
    timestamp = datetime.timestamp(datetime.now())
    save_rd_matrix(kt_matrix, timestamp)
    save_kernel(
        kernel.kernel, kernel.activator.kernel, kernel.inhibitor.kernel, timestamp
    )
    save_fourier(kernel.fourier, timestamp)
    update_csv(timestamp, kernel.activator, kernel.inhibitor, kernel.integrate())


def update_csv(timestamp, activator, inhibitor, kernel_integral):
    with open(os.path.join("images", "info.csv"), "a") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(
            [
                timestamp,
                activator.amplitude,
                activator.distance,
                activator.width,
                inhibitor.amplitude,
                inhibitor.distance,
                inhibitor.width,
                f"{kernel_integral:.3f}",
            ]
        )


def save_rd_matrix(kt_matrix, timestamp):
    fig, ax = plt.subplots()
    ax.imshow(
        np.reshape(kt_matrix, (c.MATRIX_SIZE, c.MATRIX_SIZE)), interpolation="none"
    )
    ax.set_title("Reaction Diffusion Result")
    fig.savefig(os.path.join("images", f"{timestamp}_reaction_diffusion_result.png"))
    plt.close(fig)


def save_kernel(kernel, activator, inhibitor, timestamp):
    x = np.linspace(0, 20)
    fig, ax = plt.subplots()
    ax.set_title("Kernel (Activator + Inhibitor)")
    ax.plot(x, kernel, label="Kernel")
    ax.plot(x, activator, label="Activator", linestyle="dashed")
    ax.plot(x, inhibitor, label="Inhibitor", linestyle="dashed")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, 20)
    fig.savefig(os.path.join("images", f"{timestamp}_kernel.png"))
    plt.close(fig)


def save_fourier(fourier, timestamp):
    fig, ax = plt.subplots()
    ax.set_title("Fourier Transform of the Kernel")
    ax.plot(fourier)
    ax.set_xlim(0, 20)
    ax.grid(True)
    fig.savefig(os.path.join("images", f"{timestamp}_fourier.png"))
    plt.close(fig)
