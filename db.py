"""Provides helper methods for interacting with the db. If run directly it allows the user to delete rows from the db"""

import sqlite3
import os

from kernel_helpers import Kernel


def write_kernel(kernel: Kernel) -> int:
    """Write the current kernel to the db, and return the ID of the row"""

    # Establish connection to the db
    # Assume that we are running from the project root
    with sqlite3.connect(os.path.join("images", "info.db")) as connection:
        cursor = connection.cursor()

        params = [
            kernel.activator.amplitude,
            kernel.activator.distance,
            kernel.activator.width,
            kernel.inhibitor.amplitude,
            kernel.inhibitor.distance,
            kernel.inhibitor.width,
        ]

        cursor.execute(
            """INSERT INTO info(activator_amplitude, activator_distance, activator_width, inhibitor_amplitude, inhibitor_distance, inhibitor_width)
            VALUES (?, ?, ?, ?, ?, ?)""",
            params,
        )
        connection.commit()

        return cursor.lastrowid


def delete(id: int):
    """Delete a row from the table with the given ID"""

    with sqlite3.connect(os.path.join("images", "info.db")) as connection:
        cursor = connection.cursor()

        cursor.execute("DELETE FROM info WHERE id=?", (id,))
        connection.commit()


if __name__ == "__main__":
    try:
        id = int(input("Enter the row ID to delete: "))
        delete(id)
    except ValueError as e:
        print("Invalid Input")
