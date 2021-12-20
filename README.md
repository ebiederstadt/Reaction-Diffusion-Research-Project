# Reaction Diffusion Research Project

![Code Quality](https://github.com/ebiederstadt/Reaction-Diffusion-Research-Project/actions/workflows/main.yml/badge.svg)

The project is built to simulate plant patterning using kernel based reaction diffusion techniques.

For more information on kernel based reaction diffusion, you can read the paper by Kondo: https://www.sciencedirect.com/science/article/pii/S0022519316303630. You can also read my research report.

## Requirements 

- Python (3.7-3.9), tested with python 3.9 only
- Poetry: https://python-poetry.org/docs/

After cloning the repository, run `poetry init`.

## Usage

Available Programs:

- `main.py`: classical kernel based reaction diffusion stimulation. Adjust the kernel parameters either by entering them into the textboxes or using the sliders. Saved figures will be stored in a folder matching their ID in the database (`info.db`)
- `multispecies_interactions.py`: Simulation of two interacting species using kernel based reaction diffusion. Select from a number of predetermined kernels in the `edit` menu. Switch the species shown in the bottom right using the `view` menu. Figures will be saved in a folder named `multispecies_${id}`, where `${id}` matches the ID in the `info` table, contained in `multispecies.db`.
- `db.py`: Delete items from the database if they are no longer needed
