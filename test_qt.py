from PyQt5.QtWidgets import QMessageBox, QWidget
import numpy as np
import sys
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.backends.qt_compat import QtWidgets
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import numpy as np

from kernel_helpers import Kernel
import constants as c


class Textbox_Demo(QtWidgets.QMainWindow):
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


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = Textbox_Demo()
    app.show()
    qapp.exec_()
