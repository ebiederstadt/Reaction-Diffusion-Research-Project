from enum import Enum
import os
import sys
from time import perf_counter
from typing import Optional

import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from numba.core.targetconfig import Option
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QMainWindow,
    QPushButton,
)

import constants as c
from db import write_kernel_multispecies
from interval import SetInterval
from kernel_helpers import Kernel, ConstantKernel, compute_stimulation


class KernelChoices(Enum):
    LALI = 0
    INVERTED_LALI = 1
    NESTED = 2
    THIN_STRIPED = 3
    THICK_STRIPED = 4


def from_string(s: str) -> KernelChoices:
    if s == "LALI Kernel":
        return KernelChoices.LALI
    elif s == "Inverted LALI Kernel":
        return KernelChoices.INVERTED_LALI
    elif s == "Nested Pattern Formation":
        return KernelChoices.NESTED
    elif s == "Thin Striped Kernel":
        return KernelChoices.THIN_STRIPED
    elif s == "Thick Striped Kernel":
        return KernelChoices.THICK_STRIPED
    else:
        raise ValueError("Invalid string")


def determine_kernel_params(choice: KernelChoices, kernel: Kernel):
    if choice == KernelChoices.LALI:
        kernel.update_activator(20.267, 0.0, 1.817)
        kernel.update_inhibitor(-2.133, 0.0, 5.835)
    elif choice == KernelChoices.INVERTED_LALI:
        kernel.update_activator(-22.4, 0.0, 2.748)
        kernel.update_inhibitor(8.0, 6.7, 1.278)
    elif choice == KernelChoices.NESTED:
        kernel.update_activator(21.085, 10.3, 0.739)
        kernel.update_inhibitor(-19.733, 8.7, 0.935)
    elif choice == KernelChoices.THIN_STRIPED:
        kernel.update_activator(13.251, 8.9, 0.886)
        kernel.update_inhibitor(-7.466, 0.0, 5.835)
    elif choice == KernelChoices.THICK_STRIPED:
        kernel.update_activator(13.908, 5.78, 2.013)
        kernel.update_inhibitor(-11.733, 11.5, 1.18)
    else:
        raise ValueError("Invalid kernel choice")


class MultiSpeciesWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multispecies Interactions")
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.addToolBar(NavigationToolbar(self.canvas, self))

        button_widget = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(button_widget)
        self.randomize_s1_button = QPushButton(self)
        self.randomize_s1_button.setText("Randomize S1")
        self.randomize_s1_button.clicked.connect(self.randomize_s1)
        button_layout.addWidget(self.randomize_s1_button)

        self.randomize_s2_button = QPushButton(self)
        self.randomize_s2_button.setText("Randomize S2")
        self.randomize_s2_button.clicked.connect(self.randomize_s2)
        button_layout.addWidget(self.randomize_s2_button)

        self.calculation_started = False
        self.calculate_button = QPushButton(self)
        self.calculate_button.setText("Begin Calculation")
        self.calculate_button.clicked.connect(self.start_or_start_calculation)
        button_layout.addWidget(self.calculate_button)

        self.save_button = QPushButton(self)
        self.save_button.setText("Save Figures")
        self.save_button.clicked.connect(self.save_figures)
        button_layout.addWidget(self.save_button)

        layout.addWidget(button_widget)

        self._create_menu()

        self.x = np.linspace(0, c.KERNEL_SIZE)

        # Interactions between s1 and the environment
        self.s1_environment = ConstantKernel(np.ones(c.KERNEL_ELEMS))

        # Interations between s2 and the environment
        self.s2_environment = ConstantKernel(np.ones(c.KERNEL_ELEMS))

        self.s1_s2 = Kernel()  # The effect of s1 on s2
        determine_kernel_params(KernelChoices.LALI, self.s1_s2)
        self.s2_s1 = Kernel()  # The effect of s2 on s1
        determine_kernel_params(KernelChoices.LALI, self.s2_s1)

        # Initially, the two species are exactly opposite to each other
        self.s1_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)
        self.s2_matrix = 1 - self.s1_matrix

        self.species_to_plot = 1

        self.fill_figure()

    def fill_figure(self):
        self.string = "label"

        self.rows = 2  # reducing rows speeds up textbox interaction
        self.cols = 2  # reducing cols speeds up textbox interaction
        self.plot_count = self.rows * self.cols
        self.gs = gridspec.GridSpec(
            self.rows,
            self.cols,
            left=0.05,
            right=1 - 0.02,
            top=1 - 0.04,
            bottom=0.10,
            wspace=0.3,
            hspace=0.4,
        )
        self.ax0 = self.fig.add_subplot(self.gs[0])
        self._plot_kt_matrices()

        self.ax1 = self.fig.add_subplot(self.gs[1])
        self.ax2 = self.fig.add_subplot(self.gs[2])
        self.ax3 = self.fig.add_subplot(self.gs[3])
        self._plot_kernel()

    def _create_menu(self):
        # First set of actions modify species 1
        s2_s1_lali_action = QAction(self)
        s2_s1_lali_action.setText("LALI Kernel")
        s2_s1_lali_action.setCheckable(True)
        s2_s1_lali_action.setChecked(True)

        s2_s1_inverted_lali_action = QAction(self)
        s2_s1_inverted_lali_action.setText("Inverted LALI Kernel")
        s2_s1_inverted_lali_action.setCheckable(True)

        s2_s1_nested_action = QAction(self)
        s2_s1_nested_action.setText("Nested Pattern Formation")
        s2_s1_nested_action.setCheckable(True)

        s2_s1_thin_stripes = QAction(self)
        s2_s1_thin_stripes.setText("Thin Striped Kernel")
        s2_s1_thin_stripes.setCheckable(True)

        s2_s1_thick_stripes = QAction(self)
        s2_s1_thick_stripes.setText("Thick Striped Kernel")
        s2_s1_thick_stripes.setCheckable(True)

        # Second set of actions modify s2
        s1_s2_lali_action = QAction(self)
        s1_s2_lali_action.setText("LALI Kernel")
        s1_s2_lali_action.setCheckable(True)
        s1_s2_lali_action.setChecked(True)

        s1_s2_inverted_lali_action = QAction(self)
        s1_s2_inverted_lali_action.setText("Inverted LALI Kernel")
        s1_s2_inverted_lali_action.setCheckable(True)

        s1_s2_nested_action = QAction(self)
        s1_s2_nested_action.setText("Nested Pattern Formation")
        s1_s2_nested_action.setCheckable(True)

        s1_s2_thin_stripes = QAction(self)
        s1_s2_thin_stripes.setText("Thin Striped Kernel")
        s1_s2_thin_stripes.setCheckable(True)

        s1_s2_thick_stripes = QAction(self)
        s1_s2_thick_stripes.setText("Thick Striped Kernel")
        s1_s2_thick_stripes.setCheckable(True)

        menu = self.menuBar()
        edit_menu = menu.addMenu("&Edit")
        s2_s1_action_group = QActionGroup(self)
        s2_s1_action_group.setExclusive(True)
        s2_s1_action_group.addAction(s2_s1_lali_action)
        s2_s1_action_group.addAction(s2_s1_inverted_lali_action)
        s2_s1_action_group.addAction(s2_s1_nested_action)
        s2_s1_action_group.addAction(s2_s1_thin_stripes)
        s2_s1_action_group.addAction(s2_s1_thick_stripes)

        s2_s1_menu = edit_menu.addMenu("Change how s2 will affect s1")
        s2_s1_menu.addAction(s2_s1_lali_action)
        s2_s1_menu.addAction(s2_s1_inverted_lali_action)
        s2_s1_menu.addAction(s2_s1_nested_action)
        s2_s1_menu.addAction(s2_s1_thin_stripes)
        s2_s1_menu.addAction(s2_s1_thick_stripes)
        s2_s1_menu.triggered.connect(
            lambda action: self.update_s2_s1(from_string(action.text()))
        )

        s1_s2_menu = edit_menu.addMenu("Change how s1 will affect s2")
        s1_s2_action_group = QActionGroup(self)
        s1_s2_action_group.setExclusive(True)
        s1_s2_action_group.addAction(s1_s2_lali_action)
        s1_s2_action_group.addAction(s1_s2_inverted_lali_action)
        s1_s2_action_group.addAction(s1_s2_nested_action)
        s1_s2_action_group.addAction(s1_s2_thin_stripes)
        s1_s2_action_group.addAction(s1_s2_thick_stripes)
        s1_s2_action_group.setExclusive(True)

        s1_s2_menu.addAction(s1_s2_lali_action)
        s1_s2_menu.addAction(s1_s2_inverted_lali_action)
        s1_s2_menu.addAction(s1_s2_nested_action)
        s1_s2_menu.addAction(s1_s2_thin_stripes)
        s1_s2_menu.addAction(s1_s2_thick_stripes)
        s1_s2_menu.triggered.connect(
            lambda action: self.update_s1_s2(from_string(action.text()))
        )

        view_s1 = QAction(self)
        view_s1.setCheckable(True)
        view_s1.setText("View Species 1")
        view_s1.setChecked(True)
        view_s1.triggered.connect(lambda _: self._update_species_display(1))

        view_s2 = QAction(self)
        view_s2.setCheckable(True)
        view_s2.setText("View Species 2")
        view_s2.triggered.connect(lambda _: self._update_species_display(2))

        view_group = QActionGroup(self)
        view_group.addAction(view_s1)
        view_group.addAction(view_s2)
        view_group.setExclusive(True)

        view_menu = menu.addMenu("View")
        view_menu.addAction(view_s1)
        view_menu.addAction(view_s2)

    def _update_species_display(self, option: int):
        self.species_to_plot = option
        self.ax3.cla()
        self._plot_species()
        self.fig.canvas.draw_idle()

    def _plot_kt_matrices(self, axis: Optional[Axes] = None) -> None:
        """Plot the KT Matrices on top of each other."""

        if not axis:
            axis = self.ax0

        # Transparency method copied from stackoverflow
        # https://stackoverflow.com/questions/10127284/overlay-imshow-plots-in-matplotlib
        color1 = colors.colorConverter.to_rgba("white")
        color2 = colors.colorConverter.to_rgba("black")

        cmap1 = colors.LinearSegmentedColormap.from_list(
            "cmap1", ["green", "blue"], 256
        )
        cmap2 = colors.LinearSegmentedColormap.from_list("cmap2", [color1, color2], 256)
        cmap2._init()
        alphas = np.linspace(0, 0.6, cmap2.N + 3)
        cmap2._lut[:, -1] = alphas

        axis.imshow(
            np.reshape(self.s1_matrix, (c.MATRIX_SIZE, c.MATRIX_SIZE)),
            interpolation="none",
            cmap=cmap1,
        )
        axis.imshow(
            np.reshape(self.s2_matrix, (c.MATRIX_SIZE, c.MATRIX_SIZE)),
            interpolation="none",
            cmap=cmap2,
        )
        axis.set_title("Reaction Diffusion Result")

        # Legend
        legend_patches = [
            patches.Patch(color="green", label="Species 1"),
            patches.Patch(color="blue", label="Species 2"),
        ]
        # put those patched as legend-handles into the legend
        axis.legend(
            handles=legend_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
        )

    def _plot_kernel(
        self, kernel_axis: Optional[Axes] = None, fourier_axis: Optional[Axes] = None
    ) -> None:
        """Plot the kernel and the fourier transform of the kernel"""

        if not kernel_axis:
            kernel_axis = self.ax1
        if not fourier_axis:
            fourier_axis = self.ax2

        kernel_axis.set_title("Kernel (Activator + Inhibitor)")
        kernel_axis.plot(
            self.x, self.s1_environment.kernel, label="s1 and the environment"
        )
        kernel_axis.plot(
            self.x, self.s2_environment.kernel, label="s2 and the environment"
        )
        kernel_axis.plot(self.x, self.s2_s1.kernel, label="Effect of s2 on s1")
        kernel_axis.plot(self.x, self.s1_s2.kernel, label="Effect of s1 on s2")
        kernel_axis.grid(True)
        kernel_axis.set_xlim(0, c.KERNEL_SIZE)
        kernel_axis.legend()
        kernel_axis.set_aspect("equal")

        fourier_axis.grid(True)
        fourier_axis.plot(self.x, self.s1_s2.fourier, label="Effect of s2 on s1")
        fourier_axis.plot(self.x, self.s2_s1.fourier, label="Effect of s1 on s2")
        fourier_axis.set_title("Fourier Transform of Kernels")
        fourier_axis.set_xlim(0, c.KERNEL_SIZE)
        fourier_axis.legend()

        self._plot_species()

    def _plot_species(self, axis: Optional[Axes] = None, id: Optional[int] = None):
        if not axis:
            axis = self.ax3
        if not id:
            id = self.species_to_plot

        axis.set_title(f"Species {id}")
        if id == 1:
            axis.imshow(
                np.reshape(self.s1_matrix, (200, 200)),
                interpolation=None,
                cmap="Greens",
            )
        else:
            axis.imshow(
                np.reshape(self.s2_matrix, (200, 200)), interpolation=None, cmap="Blues"
            )

    def update_s2_s1(self, choice: KernelChoices):
        """This will change how s1 is affected by s2"""

        determine_kernel_params(choice, self.s2_s1)
        self.ax1.cla()
        self.ax2.cla()
        self._plot_kernel()
        self.fig.canvas.draw_idle()

    def update_s1_s2(self, choice: KernelChoices):
        """This will change how s2 is affected by s2"""

        determine_kernel_params(choice, self.s1_s2)
        self.ax1.cla()
        self.ax2.cla()
        self._plot_kernel()
        self.fig.canvas.draw_idle()

    def randomize_s1(self):
        self.s1_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)

        self.ax0.cla()
        self.ax3.cla()
        self._plot_kt_matrices()
        self._plot_species()
        self.fig.canvas.draw_idle()

    def randomize_s2(self):
        self.s2_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)

        self.ax0.cla()
        self.ax0.cla()
        self._plot_kt_matrices()
        self._plot_species()
        self.fig.canvas.draw_idle()

    def start_or_start_calculation(self):
        if self.calculate_button.text() == "Begin Calculation":
            self.calculate_button.setText("Stop Calculation")
            self.interval = SetInterval(0.5, self.calculate_stimulation_received)
        else:
            self.calculate_button.setText("Begin Calculation")
            self.interval.cancel()

    def calculate_stimulation_received(self):
        start = perf_counter()
        s1_stimulation_matrix = compute_stimulation(self.s2_s1.cache, self.s2_matrix)
        np.clip(
            s1_stimulation_matrix,
            c.MIN_STIMULATION,
            c.MAX_STIMULATION,
            out=s1_stimulation_matrix,
        )
        s2_stimulation_matrix = compute_stimulation(self.s1_s2.cache, self.s1_matrix)
        np.clip(
            s2_stimulation_matrix,
            c.MIN_STIMULATION,
            c.MAX_STIMULATION,
            out=s2_stimulation_matrix,
        )
        self.s1_matrix = self.s1_matrix * c.DECAY_RATE + s1_stimulation_matrix / 100
        self.s2_matrix = self.s2_matrix * c.DECAY_RATE + s2_stimulation_matrix / 100
        end = perf_counter()
        print(f"Simulation took {end - start} seconds")

        self.ax0.cla()
        self.ax3.cla()
        self._plot_kt_matrices()
        self._plot_species()
        self.fig.canvas.draw_idle()

    def save_figures(self):
        id = write_kernel_multispecies(self.s2_s1, self.s1_s2)
        path = os.path.join("images", f"multispecies_{id}")
        os.mkdir(path)

        fig, ax = plt.subplots()
        self._plot_kt_matrices(ax)
        fig.savefig(os.path.join(path, "reaction_diffusion_result.png"))
        plt.close(fig)

        kernel_fig, kernel_axis = plt.subplots()
        fourier_fig, fourier_axis = plt.subplots()
        self._plot_kernel(kernel_axis, fourier_axis)
        kernel_fig.savefig(os.path.join(path, "kernels.png"))
        plt.close(kernel_fig)
        fourier_fig.savefig(os.path.join(path, "fourier.png"))
        plt.close(fourier_fig)

        species1_fig, species1_ax = plt.subplots()
        self._plot_species(species1_ax, 1)
        species1_fig.savefig(os.path.join(path, "species_1.png"))
        plt.close(species1_fig)
        species2_fig, species2_ax = plt.subplots()
        self._plot_species(species2_ax, 2)
        species2_fig.savefig(os.path.join(path, f"species_2.png"))
        plt.close(species2_fig)


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = MultiSpeciesWindow()
    app.show()
    qapp.exec_()
