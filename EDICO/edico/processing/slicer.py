"""
slicer.py
Divides the 3D mesh point cloud into horizontal z-axis slabs (slices),
assigns each point to its slice, and determines whether each point
is "mapped" or "unmapped" based on proximity to catheter contact points.


"""

import numpy as np
from dataclasses import dataclass


PADDING_MM = 5  # padding added around the mesh bounding box on each side


@dataclass
class VolumeGeometry:
    """Bounding box and grid dimensions for the rasterized volume."""
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    number_of_slices: int
    figure_width: int   # pixels
    figure_height: int  # pixels


def compute_geometry(vertices: np.ndarray, pixel_size: float, slice_thickness: float) -> VolumeGeometry:
    """
    Compute the bounding box and grid dimensions for the volume.

    """
    min_x = vertices[:, 0].min() - PADDING_MM
    min_y = vertices[:, 1].min() - PADDING_MM
    min_z = vertices[:, 2].min() - PADDING_MM

    max_x = vertices[:, 0].max() + PADDING_MM
    max_y = vertices[:, 1].max() + PADDING_MM
    max_z = vertices[:, 2].max() + PADDING_MM

    number_of_slices = int(np.floor((max_z - min_z) / slice_thickness) + 1)
    figure_width  = int(np.floor((max_x - min_x) / pixel_size) + 1)
    figure_height = int(np.floor((max_y - min_y) / pixel_size) + 1)

    return VolumeGeometry(
        min_x=min_x, min_y=min_y, min_z=min_z,
        max_x=max_x, max_y=max_y, max_z=max_z,
        number_of_slices=number_of_slices,
        figure_width=figure_width,
        figure_height=figure_height,
    )


def assign_points_to_slices(
    vertices: np.ndarray,
    voltage: np.ndarray,
    x_car: np.ndarray,
    y_car: np.ndarray,
    z_car: np.ndarray,
    geom: VolumeGeometry,
    slice_thickness: float,
    fill_threshold: float,
) -> tuple:
    """
    Assign each vertex to its z-slice and determine its mapped/unmapped status.

    """
    n_points = len(vertices)
    n_slices = geom.number_of_slices

    slice_data = np.zeros((n_slices, n_points, 3))
    color_data = np.zeros((n_slices, n_points))

    for slice_index in range(n_slices):
        slice_z_start = geom.min_z + slice_index * slice_thickness
        slice_z_end   = slice_z_start + slice_thickness

        # Boolean mask: which vertices fall in this slice?
        in_slice = (vertices[:, 2] >= slice_z_start) & (vertices[:, 2] < slice_z_end)
        indices  = np.where(in_slice)[0]

        for i in indices:
            x_val = vertices[i, 0]
            y_val = vertices[i, 1]
            z_val = vertices[i, 2]

            slice_data[slice_index, i, 0] = x_val
            slice_data[slice_index, i, 1] = y_val
            slice_data[slice_index, i, 2] = z_val

            # Is this point within fill_threshold of any catheter contact?
            distances = np.sqrt(
                (x_val - x_car) ** 2 +
                (y_val - y_car) ** 2 +
                (z_val - z_car) ** 2
            )
            is_within_range = np.any(distances <= fill_threshold)

            # OPEN QUESTION: voltage > 0 check discards zero-voltage points.
            # Near-zero bipolar voltage indicates scar tissue and may be
            # clinically meaningful. Confirm with clinical team whether
            # voltage == 0 is a sentinel (no data) or a valid measurement.
            if is_within_range and voltage[i] > 0:
                color_data[slice_index, i] = voltage[i] * 1000
                # OPEN QUESTION: the × 1000 factor assumes raw values are
                # in Volts and converts to mV. Verify with the clinical team.
            else:
                color_data[slice_index, i] = -500  # unmapped sentinel

    return slice_data, color_data