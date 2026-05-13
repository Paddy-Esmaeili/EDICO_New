"""
coordinates.py
Rotates mesh vertices and CAR catheter points from the mapping system's
native coordinate system into DICOM head-first supine (HFS) orientation.

The rotation applied is:
    x_new =  -z_old
    y_new =   x_old
    z_new =   y_old

OPEN QUESTION: This rotation is hardcoded and was empirically derived
from one dataset. If the mapping system exports data in a different
orientation (e.g. different patient positioning), this rotation will
produce silently incorrect output. The 8-line mesh header (currently
discarded in mesh_reader.py) may contain the source coordinate system
and should be investigated.
"""

import numpy as np


def rotate_to_hfs(vertices: np.ndarray, car_array: np.ndarray):
    """
    Rotate mesh vertices and catheter points to DICOM HFS orientation.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Vertex coordinates in the mapping system's native coordinate system.
    car_array : np.ndarray, shape (M, 3)
        Catheter contact point coordinates in the same native system.

    Returns
    -------
    vertices_rotated : np.ndarray, shape (N, 3)
        Rotated vertex coordinates.
    x_car : np.ndarray, shape (M,)
    y_car : np.ndarray, shape (M,)
    z_car : np.ndarray, shape (M,)
        Rotated catheter coordinates as separate 1D arrays,
        ready for distance calculations in slicer.py.
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