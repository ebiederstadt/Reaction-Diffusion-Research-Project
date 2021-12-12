import sys
from time import perf_counter

import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QPushButton

import constants as c
from interval import SetInterval
from kernel_helpers import Kernel, ConstantKernel, compute_stimulation


class KTMethod(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KT Method Simulator")
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.addToolBar(NavigationToolbar(self.canvas, self))

        self._textwidget = QtWidgets.QWidget()
        textlayout = QtWidgets.QHBoxLayout(self._textwidget)
        self.activator_textbox = QtWidgets.QLineEdit(self)
        self.activator_textbox.editingFinished.connect(self.on_activator_submit)
        textlayout.addWidget(QtWidgets.QLabel("A(x) (Amplitude, Width, Distance): "))
        textlayout.addWidget(self.activator_textbox)

        self.inhibitor_textbox = QtWidgets.QLineEdit(self)
        self.inhibitor_textbox.editingFinished.connect(self.on_inhibitor_submit)
        textlayout.addWidget(QtWidgets.QLabel("I(x) (Amplitude, Width, Distance): "))
        textlayout.addWidget(self.inhibitor_textbox)

        layout.addWidget(self._textwidget)

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

        self.x = np.linspace(0, c.KERNEL_SIZE)

        # Interactions between s1 and the environment
        self.s1_environment = ConstantKernel(np.ones(c.KERNEL_ELEMS))

        # Interations between s2 and the environment
        self.s2_environment = ConstantKernel(np.ones(c.KERNEL_ELEMS))

        self.s1_s2 = Kernel()  # The effect of s1 on s2
        self.s2_s1 = Kernel()  # The effect of s2 on s1

        # Initially, the two species are exactly opposite to each other
        self.s1_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)
        self.s2_matrix = 1 - self.s1_matrix

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
        self.ax1.set_title("Kernel Functions")
        self.ax1.grid(True)
        self.ax1.plot(self.s1_environment.kernel, label="s1 and the environment")
        self.ax1.plot(self.s2_environment.kernel, label="s2 and the environment")
        self.ax1.plot(self.s2_s1.kernel, label="The effect of s2 on s1")
        self.ax1.plot(self.s1_s2.kernel, label="The effect of s1 on s2")
        self.ax1.set_xlim(0, c.KERNEL_SIZE)
        self.ax1.legend()
        self.ax1.set_aspect("equal")

        self.ax2 = self.fig.add_subplot(self.gs[2])
        self.ax2.grid(True)
        self.ax2.plot(self.s1_environment.fourier)
        self.ax2.set_title("Fourier Transform of Kernel")
        self.ax2.set_xlim(0, c.KERNEL_SIZE)

    def _plot_kt_matrices(self):
        """Plot the KT Matrices on top of each other."""

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

        self.ax0.imshow(
            np.reshape(self.s1_matrix, (c.MATRIX_SIZE, c.MATRIX_SIZE)),
            interpolation="none",
            cmap=cmap1,
        )
        im = self.ax0.imshow(
            np.reshape(self.s2_matrix, (c.MATRIX_SIZE, c.MATRIX_SIZE)),
            interpolation="none",
            cmap=cmap2,
        )
        self.ax0.set_title("Reaction Diffusion Result")

        # Legend
        legend_patches = [
            patches.Patch(color="green", label="Species 1"),
            patches.Patch(color="blue", label="Species 2"),
        ]
        # put those patched as legend-handles into the legend
        self.ax0.legend(
            handles=legend_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
        )

    def on_activator_submit(self):
        text = self.activator_textbox.text()
        try:
            amplitude, width, distance = [float(x) for x in text.split(",")]
            if self.s1_environment.activator.diff(amplitude, width, distance):
                self.s1_environment.update_activator(amplitude, distance, width)
                # Update the kernel graph
                self.ax1.cla()
                self.ax1.set_title("Kernel (Activator + Inhibitor)")
                self.ax1.grid(True)
                self.ax1.plot(self.x, self.s1_environment.kernel, label="Kernel")
                self.ax1.plot(
                    self.x,
                    self.s1_environment.activator.kernel,
                    label="Activator",
                    linestyle="dashed",
                )
                self.ax1.plot(
                    self.x,
                    self.s1_environment.inhibitor.kernel,
                    label="Inhibitor",
                    linestyle="dashed",
                )
                self.ax1.set_xlim(0, c.KERNEL_SIZE)
                self.ax1.legend()
                self.ax1.set_aspect("equal")
                # Update the fourier transform
                self.ax2.cla()
                self.ax2.grid(True)
                self.ax2.plot(self.s1_environment.fourier)
                self.ax2.set_title("Fourier Transform of Kernel")
                self.ax2.set_xlim(0, c.KERNEL_SIZE)

                self.fig.canvas.draw_idle()
        except ValueError:
            msgbox = QMessageBox(self)
            msgbox.setText(f"Invalid Activator Input: {text}")
            msgbox.exec()

    def on_inhibitor_submit(self):
        text = self.inhibitor_textbox.text()
        try:
            amplitude, width, distance = [float(x) for x in text.split(",")]
            if self.s1_environment.inhibitor.diff(amplitude, distance, width):
                self.s1_environment.update_inhibitor(amplitude, distance, width)
                # Update the kernel graph
                self.ax1.cla()
                self.ax1.set_title("Kernel (Activator + Inhibitor)")
                self.ax1.grid(True)
                self.ax1.plot(self.x, self.s1_environment.kernel, label="Kernel")
                self.ax1.plot(
                    self.x,
                    self.s1_environment.activator.kernel,
                    label="Activator",
                    linestyle="dashed",
                )
                self.ax1.plot(
                    self.x,
                    self.s1_environment.inhibitor.kernel,
                    label="Inhibitor",
                    linestyle="dashed",
                )
                self.ax1.set_xlim(0, c.KERNEL_SIZE)
                self.ax1.legend()
                self.ax1.set_aspect("equal")
                # Update the fourier transform
                self.ax2.cla()
                self.ax2.grid(True)
                self.ax2.plot(self.s1_environment.fourier)
                self.ax2.set_title("Fourier Transform of Kernel")
                self.ax2.set_xlim(0, c.KERNEL_SIZE)

                self.fig.canvas.draw_idle()
        except ValueError:
            msgbox = QMessageBox(self)
            msgbox.setText(f"Invalid Inhibitor Input: {text}")
            msgbox.exec()

    def randomize_s1(self):
        self.s1_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)

        self.ax0.cla()
        self._plot_kt_matrices()
        self.fig.canvas.draw_idle()

    def randomize_s2(self):
        self.s2_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)

        self.ax0.cla()
        self._plot_kt_matrices()
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
        stimulation_matrix = compute_stimulation(
            self.s1_environment.cache, self.s1_matrix
        )
        np.clip(
            stimulation_matrix,
            c.MIN_STIMULATION,
            c.MAX_STIMULATION,
            out=stimulation_matrix,
        )
        print(
            f"stimulation max: {stimulation_matrix.max()} min: {stimulation_matrix.min()}, mean: {stimulation_matrix.mean()}"
        )
        self.s1_matrix = self.s1_matrix * c.DECAY_RATE + stimulation_matrix / 100
        end = perf_counter()
        print(f"Simulation took {end - start} seconds")

        self.ax0.cla()
        self._plot_kt_matrices()
        self.fig.canvas.draw_idle()

    def save_figures(self):
        print("FIXME: Implement Save Figures!")


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = KTMethod()
    app.show()
    qapp.exec_()
