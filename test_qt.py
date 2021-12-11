import sys
from time import perf_counter

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QPushButton

import constants as c
from interval import SetInterval
from kernel_helpers import Kernel, compute_stimulation


class KTMethod(QMainWindow):
    def __init__(self):
        super().__init__()
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
        self.randomize_button = QPushButton(self)
        self.randomize_button.setText("Randomize Matrix")
        self.randomize_button.clicked.connect(self.randomize_matrix)
        button_layout.addWidget(self.randomize_button)

        self.calculation_started = False
        self.calculate_button = QPushButton(self)
        self.calculate_button.setText("Begin Calculation")
        self.calculate_button.clicked.connect(self.start_or_start_calculation)
        button_layout.addWidget(self.calculate_button)

        layout.addWidget(button_widget)

        self.kernel = Kernel()
        self.kt_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)

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
        self.ax0.imshow(np.reshape(self.kt_matrix, (200, 200)), interpolation="none")
        self.ax0.set_title("Reaction Diffusion Result")

        self.ax1 = self.fig.add_subplot(self.gs[1])
        self.ax1.set_title("Kernel (Activator + Inhibitor)")
        self.ax1.grid(True)
        self.ax1.plot(self.kernel.kernel, label="Kernel")
        self.ax1.plot(
            self.kernel.activator.kernel, label="Activator", linestyle="dashed"
        )
        self.ax1.plot(
            self.kernel.inhibitor.kernel, label="Inhibitor", linestyle="dashed"
        )
        self.ax1.set_xlim(0, c.KERNEL_SIZE)
        self.ax1.legend()
        self.ax1.set_aspect("equal")

        self.ax2 = self.fig.add_subplot(self.gs[2])
        self.ax2.grid(True)
        self.ax2.plot(self.kernel.fourier)
        self.ax2.set_title("Fourier Transform of Kernel")
        self.ax2.set_xlim(0, c.KERNEL_SIZE)

    def on_activator_submit(self):
        text = self.activator_textbox.text()
        try:
            amplitude, width, distance = [float(x) for x in text.split(",")]
            if self.kernel.activator.diff(amplitude, width, distance):
                self.kernel.update_activator(amplitude, distance, width)
                # Update the kernel graph
                self.ax1.cla()
                self.ax1.set_title("Kernel (Activator + Inhibitor)")
                self.ax1.grid(True)
                self.ax1.plot(self.kernel.kernel, label="Kernel")
                self.ax1.plot(
                    self.kernel.activator.kernel, label="Activator", linestyle="dashed"
                )
                self.ax1.plot(
                    self.kernel.inhibitor.kernel, label="Inhibitor", linestyle="dashed"
                )
                self.ax1.set_xlim(0, c.KERNEL_SIZE)
                self.ax1.legend()
                self.ax1.set_aspect("equal")
                # Update the fourier transform
                self.ax2.cla()
                self.ax2.grid(True)
                self.ax2.plot(self.kernel.fourier)
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
            if self.kernel.inhibitor.diff(amplitude, distance, width):
                self.kernel.update_inhibitor(amplitude, distance, width)
                # Update the kernel graph
                self.ax1.cla()
                self.ax1.set_title("Kernel (Activator + Inhibitor)")
                self.ax1.grid(True)
                self.ax1.plot(self.kernel.kernel, label="Kernel")
                self.ax1.plot(
                    self.kernel.activator.kernel, label="Activator", linestyle="dashed"
                )
                self.ax1.plot(
                    self.kernel.inhibitor.kernel, label="Inhibitor", linestyle="dashed"
                )
                self.ax1.set_xlim(0, c.KERNEL_SIZE)
                self.ax1.legend()
                self.ax1.set_aspect("equal")
                # Update the fourier transform
                self.ax2.cla()
                self.ax2.grid(True)
                self.ax2.plot(self.kernel.fourier)
                self.ax2.set_title("Fourier Transform of Kernel")
                self.ax2.set_xlim(0, c.KERNEL_SIZE)

                self.fig.canvas.draw_idle()
        except ValueError:
            msgbox = QMessageBox(self)
            msgbox.setText(f"Invalid Inhibitor Input: {text}")
            msgbox.exec()

    def randomize_matrix(self):
        self.kt_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)

        self.ax0.cla()
        self.ax0.imshow(np.reshape(self.kt_matrix, (200, 200)), interpolation="none")
        self.ax0.set_title("Reaction Diffusion Result")
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
        stimulation_matrix = compute_stimulation(self.kernel.cache, self.kt_matrix)
        np.clip(
            stimulation_matrix,
            c.MIN_STIMULATION,
            c.MAX_STIMULATION,
            out=stimulation_matrix,
        )
        print(
            f"stimulation max: {stimulation_matrix.max()} min: {stimulation_matrix.min()}, mean: {stimulation_matrix.mean()}"
        )
        self.kt_matrix = self.kt_matrix * c.DECAY_RATE + stimulation_matrix / 100
        end = perf_counter()
        print(f"Simulation took {end - start} seconds")

        self.ax0.cla()
        self.ax0.imshow(np.reshape(self.kt_matrix, (200, 200)), interpolation="none")
        self.ax0.set_title("Reaction Diffusion Result")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = KTMethod()
    app.show()
    qapp.exec_()
