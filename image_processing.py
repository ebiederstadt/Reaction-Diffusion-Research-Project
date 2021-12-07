import os
import csv
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

from kernel_helpers import integrate_kernel


def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)


def save_fig(filename: str, fig, ax):
    extent = full_extent(ax).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join("images", filename), bbox_inches=extent)


def write_figures(kt_matrix, kernel, activator, inhibitor):
    timestamp = datetime.timestamp(datetime.now())
    save_rd_matrix(kt_matrix, timestamp)
    save_kernel(kernel, activator.kernel, inhibitor.kernel, timestamp)
    update_csv(timestamp, activator, inhibitor, integrate_kernel(kernel))


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
    ax.imshow(np.reshape(kt_matrix, (200, 200)), interpolation="none")
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
