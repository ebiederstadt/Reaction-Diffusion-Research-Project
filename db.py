"""Provides helper methods for interacting with the db. If run directly it allows the user to delete rows from the db"""

import sqlite3
import os
import argparse

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
            kernel.compute_2d_integral(),
        ]

        cursor.execute(
            """INSERT INTO info(activator_amplitude, activator_distance, activator_width, inhibitor_amplitude, inhibitor_distance, inhibitor_width, integral_2d)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            params,
        )
        connection.commit()

        return cursor.lastrowid


def write_kernel_multispecies(s1_kernel: Kernel, s2_kernel: Kernel) -> int:
    """Write both kernels in the multispecies case

    :params
    s1_kernel: Measures the effect s2 has on s1
    s2_kernel: Measures the effect s1 has on s2
    """

    with sqlite3.connect(os.path.join("images", "multispecies.db")) as connection:
        cursor = connection.cursor()

        def create_params(kernel, id):
            return [
                str(id),
                kernel.activator.amplitude,
                kernel.activator.distance,
                kernel.activator.width,
                kernel.inhibitor.amplitude,
                kernel.inhibitor.distance,
                kernel.inhibitor.width,
                kernel.compute_2d_integral(),
            ]

        # Create the ID in the info table
        cursor.execute("INSERT INTO info DEFAULT VALUES")
        id = cursor.lastrowid

        # Write s1 kernel
        s1_params = create_params(s1_kernel, id)
        cursor.execute(
            """INSERT INTO s1_info(info_id, activator_amplitude, activator_distance, activator_width, inhibitor_amplitude, inhibitor_distance, inhibitor_width, integral_2d)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            s1_params,
        )

        # Write the s2 kernel
        s2_params = create_params(s2_kernel, id)
        cursor.execute(
            """INSERT INTO s2_info(info_id, activator_amplitude, activator_distance, activator_width, inhibitor_amplitude, inhibitor_distance, inhibitor_width, integral_2d)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            s2_params,
        )

        connection.commit()

        return id


def delete(id: int):
    """Delete a row from the table with the given ID"""

    with sqlite3.connect(os.path.join("images", "info.db")) as connection:
        cursor = connection.cursor()

        cursor.execute("DELETE FROM info WHERE id=?", (id,))
        connection.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete one row from the DB's")
    parser.add_argument(
        "db",
        type=str,
        help="DB to delete from. Options are single or multi (single species or multispecies)",
    )
    args = parser.parse_args()
    try:
        id = int(input("Enter the row ID to delete: "))
    except ValueError:
        print("Invalid input")
    if args.db.lower() == "single":
        delete(id)
    elif args.db.lower() == "multi":
        print("FIXME")
    else:
        print("Invalid Input")
