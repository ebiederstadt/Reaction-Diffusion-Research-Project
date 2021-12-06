import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.transforms import Bbox
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import numpy.matlib

from image_processing import save_fig

# Constants
MATRIX_SIZE = 200
KERNEL_SIZE = 20

x = np.linspace(0, KERNEL_SIZE)


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


def save_figures(event):
    """Save all the figures"""
    time_stamp = datetime.timestamp(datetime.now())
    save_fig(f"{time_stamp}_reaction_diffusion_result.png", fig, ax[0][0])
    save_fig(f"{time_stamp}_kernel.png", fig, ax[0][1])
    save_fig(f"{time_stamp}_fourier_transform.png", fig, ax[1][0])


# Starting values for the kernel and the simulation matrix
kt_matrix = np.matlib.rand(MATRIX_SIZE, MATRIX_SIZE)
x = np.linspace(0, KERNEL_SIZE)
activator = KernelPortion(21, 0, 2.06)
inhibitor = KernelPortion(-6.5, 4.25, 1.38)

kernel = activator.kernel + inhibitor.kernel

fig, ax = plt.subplots(2, 2)
ax[0][0].set_title("Reaction Diffusion Result")
ax[0][0].imshow(kt_matrix)

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

plt.show()
