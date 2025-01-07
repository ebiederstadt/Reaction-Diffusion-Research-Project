from pathlib import Path
import json
import logging
from argparse import ArgumentParser
import sys

from matplotlib.backends.qt_compat import QtWidgets

from kt_rd.singlespecies_interactions import KTMethod
from kt_rd.multispecies_interactions import MultiSpeciesWindow


def setup_logging():
    config_file = Path("logging_configs/config.json")
    with open(config_file) as file:
        config = json.load(file)
    logging.config.dictConfig(config)


if __name__ == "__main__":
    setup_logging()

    parser = ArgumentParser(
        description="Simulate natural pattern formation using a kernel based reaction diffusion system"
    )
    parser.add_argument(
        "-s",
        "--single_species",
        action="store_true",
        default=True,
        help="Display the single-species interaction window (default)",
    )
    parser.add_argument(
        "-m",
        "--multi_species",
        action="store_true",
        help="Display the multi-species interaction window.",
    )

    args = parser.parse_args()
    if not args.multi_species:
        qapp = QtWidgets.QApplication(sys.argv)
        app = KTMethod()
        app.show()
        qapp.exec_()

    else:
        qapp = QtWidgets.QApplication(sys.argv)
        app = MultiSpeciesWindow()
        app.show()
        qapp.exec_()
