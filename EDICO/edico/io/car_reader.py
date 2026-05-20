"""
car_reader.py
Reads the _car file exported by the electroanatomic mapping system.
"""

import numpy as np


def parse_car(filepath: str) -> np.ndarray:
    """
    Parse a _car file and return catheter contact point coordinates.

    """
    car_values = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            if not line.startswith("P"):
                continue
            parts = line.strip().split()
            try:
                values = list(map(float, parts[4:7]))
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"Could not parse coordinates from line {line_num} in {filepath}: "
                    f"{repr(line)}"
                ) from e

            if len(values) < 3:
                raise ValueError(
                    f"Line {line_num} in {filepath} has fewer than 7 columns. "
                    f"Expected at least 7 (index 4-6 are x, y, z)."
                )

            car_values.append(values)

    if len(car_values) == 0:
        raise ValueError(
            f"No catheter contact points ('P' lines) found in {filepath}. "
            f"Check that the correct _car file was selected."
        )

    return np.array(car_values)