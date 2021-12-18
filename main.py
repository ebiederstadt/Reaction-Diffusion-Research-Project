from enum import Enum
import sys
from time import perf_counter

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)
from PyQt5.QtCore import Qt

import constants as c
from interval import SetInterval
from kernel_helpers import Kernel, compute_stimulation
from image_processing import write_figures
from widgets import DoubleSlider


class UpdateOptions(Enum):
    AMPLITUDE = 0
    WIDTH = 1
    DISTANCE = 3


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

        text_widget = QWidget()
        textlayout = QHBoxLayout(text_widget)
        self.activator_textbox = QtWidgets.QLineEdit(self)
        self.activator_textbox.editingFinished.connect(self.on_activator_submit)
        textlayout.addWidget(QtWidgets.QLabel("A(x) (Amplitude, Width, Distance): "))
        textlayout.addWidget(self.activator_textbox)

        self.inhibitor_textbox = QLineEdit(self)
        self.inhibitor_textbox.editingFinished.connect(self.on_inhibitor_submit)
        textlayout.addWidget(QtWidgets.QLabel("I(x) (Amplitude, Width, Distance): "))
        textlayout.addWidget(self.inhibitor_textbox)

        layout.addWidget(text_widget)

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

        self.save_button = QPushButton(self)
        self.save_button.setText("Save Figures")
        self.save_button.clicked.connect(self.save_figures)
        button_layout.addWidget(self.save_button)

        layout.addWidget(button_widget)

        self.x = np.linspace(0, c.KERNEL_SIZE)
        self.kernel = Kernel()
        self.kt_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)

        # Sliders for changing up the kernel
        slider_widget = QWidget()
        slider_layout = QHBoxLayout(slider_widget)

        amplitude_widget = QWidget()
        amplitude_layout = QVBoxLayout(amplitude_widget)

        activator_amplitude = QLabel("Activator Amplitude")
        amplitude_layout.addWidget(activator_amplitude)

        activator_amplitude_slider = DoubleSlider(Qt.Horizontal)
        activator_amplitude_slider.setMinimum(-40)
        activator_amplitude_slider.setMaximum(40)
        activator_amplitude_slider.setValue(self.kernel.activator.amplitude)
        activator_amplitude_slider.valueChanged.connect(
            lambda: self.update_activator(
                activator_amplitude_slider.value(), UpdateOptions.AMPLITUDE
            )
        )
        activator_amplitude_slider.sliderReleased.connect(
            lambda: self.update_activator(
                activator_amplitude_slider.value(),
                UpdateOptions.AMPLITUDE,
                do_full_update=True,
            )
        )
        amplitude_layout.addWidget(activator_amplitude_slider)

        inhibitor_amplitude = QLabel("Inhibitor Amplitude")
        amplitude_layout.addWidget(inhibitor_amplitude)

        inhibitor_amplitude_slider = DoubleSlider(Qt.Horizontal)
        inhibitor_amplitude_slider.setMinimum(-40)
        inhibitor_amplitude_slider.setMaximum(40)
        inhibitor_amplitude_slider.setValue(self.kernel.inhibitor.amplitude)
        inhibitor_amplitude_slider.valueChanged.connect(
            lambda: self.update_inhibitor(
                inhibitor_amplitude_slider.value(), UpdateOptions.AMPLITUDE
            )
        )
        inhibitor_amplitude_slider.sliderReleased.connect(
            lambda: self.update_inhibitor(
                inhibitor_amplitude_slider.value(),
                UpdateOptions.AMPLITUDE,
                do_full_update=True,
            )
        )
        amplitude_layout.addWidget(inhibitor_amplitude_slider)

        slider_layout.addWidget(amplitude_widget)

        width_widget = QWidget()
        width_layout = QVBoxLayout(width_widget)

        activator_width = QLabel("Activator Width")
        width_layout.addWidget(activator_width)

        activator_width_slider = DoubleSlider(Qt.Horizontal)
        activator_width_slider.setMinimum(0)
        activator_width_slider.setMaximum(10)
        activator_width_slider.setValue(self.kernel.activator.width)
        activator_width_slider.valueChanged.connect(
            lambda: self.update_activator(
                activator_width_slider.value(), UpdateOptions.WIDTH
            )
        )
        activator_width_slider.sliderReleased.connect(
            lambda: self.update_activator(
                activator_width_slider.value(),
                UpdateOptions.WIDTH,
                do_full_update=True,
            )
        )
        width_layout.addWidget(activator_width_slider)

        inhibitor_width = QLabel("Inhibitor Width")
        width_layout.addWidget(inhibitor_width)

        inhibitor_width_slider = DoubleSlider(Qt.Horizontal)
        inhibitor_width_slider.setMinimum(0)
        inhibitor_width_slider.setMaximum(10)
        inhibitor_width_slider.setValue(self.kernel.inhibitor.width)
        inhibitor_width_slider.valueChanged.connect(
            lambda: self.update_inhibitor(
                inhibitor_width_slider.value(), UpdateOptions.WIDTH
            )
        )
        inhibitor_width_slider.sliderReleased.connect(
            lambda: self.update_inhibitor(
                inhibitor_width_slider.value(),
                UpdateOptions.WIDTH,
                do_full_update=True,
            )
        )
        width_layout.addWidget(inhibitor_width_slider)

        slider_layout.addWidget(width_widget)

        distance_widget = QWidget()
        distance_layout = QVBoxLayout(distance_widget)

        activator_distance = QLabel("Activator Distance")
        distance_layout.addWidget(activator_distance)

        activator_distance_slider = DoubleSlider(Qt.Horizontal)
        activator_distance_slider.setMinimum(0)
        activator_distance_slider.setMaximum(20)
        activator_distance_slider.setValue(self.kernel.activator.distance)
        activator_distance_slider.valueChanged.connect(
            lambda: self.update_activator(
                activator_distance_slider.value(), UpdateOptions.DISTANCE
            )
        )
        inhibitor_width_slider.sliderReleased.connect(
            lambda: self.update_activator(
                inhibitor_width_slider.value(),
                UpdateOptions.WIDTH,
                do_full_update=True,
            )
        )
        distance_layout.addWidget(activator_distance_slider)

        inhibitor_distance = QLabel("Inhibitor Distance")
        distance_layout.addWidget(inhibitor_distance)

        inhibitor_distance_slider = DoubleSlider(Qt.Horizontal)
        inhibitor_distance_slider.setMinimum(0)
        inhibitor_distance_slider.setMaximum(20)
        inhibitor_distance_slider.setValue(self.kernel.inhibitor.distance)
        inhibitor_distance_slider.valueChanged.connect(
            lambda: self.update_inhibitor(
                inhibitor_distance_slider.value(), UpdateOptions.DISTANCE
            )
        )
        inhibitor_distance_slider.sliderReleased.connect(
            lambda: self.update_inhibitor(
                inhibitor_distance_slider.value(),
                UpdateOptions.DISTANCE,
                do_full_update=True,
            )
        )
        distance_layout.addWidget(inhibitor_distance_slider)

        slider_layout.addWidget(distance_widget)

        layout.addWidget(slider_widget)

        self.fill_figure()

    def fill_figure(self):
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
        self.ax1 = self.fig.add_subplot(self.gs[1])
        self.ax2 = self.fig.add_subplot(self.gs[2])
        self._plot_kt_matrix()
        self._plot_kernel()

    def _plot_kt_matrix(self):
        self.ax0.imshow(np.reshape(self.kt_matrix, (200, 200)), interpolation="none")
        self.ax0.set_title("Reaction Diffusion Result")

    def _plot_kernel(self):
        """Plot the Kernel and the Fourier Transform of the kernel"""

        x = np.linspace(0, c.KERNEL_SIZE)
        self.ax1.set_title("Kernel (Activator + Inhibitor)")
        self.ax1.grid(True)
        self.ax1.plot(x, self.kernel.kernel, label="Kernel")
        self.ax1.plot(
            x, self.kernel.activator.kernel, label="Activator", linestyle="dashed"
        )
        self.ax1.plot(
            x, self.kernel.inhibitor.kernel, label="Inhibitor", linestyle="dashed"
        )
        self.ax1.set_xlim(0, c.KERNEL_SIZE)
        self.ax1.legend()

        self.ax2.grid(True)
        self.ax2.plot(self.kernel.fourier)
        self.ax2.set_title("Fourier Transform of Kernel")
        self.ax2.set_xlim(0, c.KERNEL_SIZE)

    def on_activator_submit(self):
        text = self.activator_textbox.text()
        try:
            amplitude, width, distance = [float(x) for x in text.split(",")]
            print(f"Update activator with params: {amplitude, distance, width}")
            if self.kernel.activator.diff(amplitude, width, distance):
                self.kernel.update_activator(amplitude, distance, width)
                self.ax1.cla()
                self.ax2.cla()
                self._plot_kernel()
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
                self.ax1.cla()
                self.ax2.cla()
                self._plot_kernel()
                self.fig.canvas.draw_idle()
        except ValueError:
            msgbox = QMessageBox(self)
            msgbox.setText(f"Invalid Inhibitor Input: {text}")
            msgbox.exec()

    def update_activator(self, value: float, what: UpdateOptions, do_full_update=False):
        amplitude = self.kernel.activator.amplitude
        width = self.kernel.activator.width
        distance = self.kernel.activator.distance

        if what == UpdateOptions.AMPLITUDE:
            amplitude = value
        elif what == UpdateOptions.WIDTH:
            width = value
        elif what == UpdateOptions.DISTANCE:
            distance = value
        else:
            raise ValueError("Invalid value when updating activator!")

        if self.kernel.activator.diff(amplitude, distance, width):
            if do_full_update:
                self.kernel.update_activator(amplitude, distance, width)
            else:
                self.kernel.partial_update_activator(amplitude, distance, width)
            self.ax1.cla()
            self.ax2.cla()
            self._plot_kernel()
            self.fig.canvas.draw_idle()

    def update_inhibitor(self, value: float, what: UpdateOptions, do_full_update=False):
        amplitude = self.kernel.inhibitor.amplitude
        width = self.kernel.inhibitor.width
        distance = self.kernel.inhibitor.distance

        if what == UpdateOptions.AMPLITUDE:
            amplitude = value
        elif what == UpdateOptions.WIDTH:
            width = value
        elif what == UpdateOptions.DISTANCE:
            distance = value
        else:
            raise ValueError("Invalid value when updating inhibitor!")

        if self.kernel.inhibitor.diff(amplitude, distance, width):
            if do_full_update:
                self.kernel.update_inhibitor(amplitude, distance, width)
            else:
                self.kernel.parital_update_inhibitor(amplitude, distance, width)
            self.ax1.cla()
            self.ax2.cla()
            self._plot_kernel()
            self.fig.canvas.draw_idle()

    def randomize_matrix(self):
        self.kt_matrix = np.random.rand(c.MATRIX_SIZE * c.MATRIX_SIZE)

        self.ax0.cla()
        self._plot_kt_matrix()
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
        self._plot_kt_matrix()
        self.fig.canvas.draw_idle()

    def save_figures(self):
        write_figures(self.kt_matrix, self.kernel)


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = KTMethod()
    app.show()
    qapp.exec_()
