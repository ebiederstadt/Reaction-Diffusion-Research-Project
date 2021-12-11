import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import constants as c
from db import write_kernel
from kernel_helpers import Kernel


def write_figures(kt_matrix: np.ndarray, kernel: Kernel):
    id = write_kernel(kernel)
    path = os.path.join("images", str(id))
    os.mkdir(path)
    save_rd_matrix(kt_matrix, path)
    save_kernel(kernel.kernel, kernel.activator.kernel, kernel.inhibitor.kernel, path)
    save_fourier(kernel.fourier, path)


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


def save_rd_matrix(kt_matrix: np.ndarray, path: str):
    resized_matrix = np.reshape(kt_matrix, (c.MATRIX_SIZE, c.MATRIX_SIZE))

    # Save a copy for presentation and showing off
    fig, ax = plt.subplots()
    ax.imshow(resized_matrix, interpolation="none")
    ax.set_title("Reaction Diffusion Result")
    fig.savefig(os.path.join(path, f"reaction_diffusion_result.png"))
    plt.close(fig)

    # Save a copy for use for heatmaps and whatnot
    plt.imsave(
        os.path.join(path, "reaction_diffusion_heatmap.png"),
        resized_matrix,
        cmap="gray",
    )


def save_kernel(
    kernel: np.ndarray, activator: np.ndarray, inhibitor: np.ndarray, path: str
):
    x = np.linspace(0, 20)
    fig, ax = plt.subplots()
    ax.set_title("Kernel (Activator + Inhibitor)")
    ax.plot(x, kernel, label="Kernel")
    ax.plot(x, activator, label="Activator", linestyle="dashed")
    ax.plot(x, inhibitor, label="Inhibitor", linestyle="dashed")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(0, 20)
    fig.savefig(os.path.join(path, "kernel.png"))
    plt.close(fig)


def save_fourier(fourier: np.ndarray, path: str):
    fig, ax = plt.subplots()
    ax.set_title("Fourier Transform of the Kernel")
    ax.plot(fourier)
    ax.set_xlim(0, 20)
    ax.grid(True)
    fig.savefig(os.path.join(path, "fourier.png"))
    plt.close(fig)
