"""
coordinates.py
Rotates mesh vertices and CAR catheter points from the mapping system's
native coordinate system into DICOM head-first supine (HFS) orientation.

"""

import numpy as np


def rotate_to_hfs(vertices: np.ndarray, car_array: np.ndarray):
    """
    Rotate mesh vertices and catheter points to DICOM HFS orientation.


    """
    # Rotate catheter points
    x_car = -car_array[:, 2]
    y_car =  car_array[:, 0]
    z_car =  car_array[:, 1]

    # Rotate vertices (copy first to avoid modifying the input array)
    v = np.copy(vertices)
    v[:, 0] = -vertices[:, 2]
    v[:, 1] =  vertices[:, 0]
    v[:, 2] =  vertices[:, 1]

    return v, x_car, y_car, z_car