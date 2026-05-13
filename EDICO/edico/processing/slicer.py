"""
slicer.py
Divides the 3D mesh point cloud into horizontal z-axis slabs (slices),
assigns each point to its slice, and determines whether each point
is "mapped" or "unmapped" based on proximity to catheter contact points.

Mapped vs unmapped logic:
    A mesh vertex is considered "mapped" if:
        1. It is within fill_threshold mm of at least one catheter contact point, AND
        2. Its voltage value is > 0

    If either condition fails, the point is labelled unmapped and assigned
    a sentinel value of -500.

OPEN QUESTIONS:
    - Should voltage == 0 be treated as unmapped (current behaviour) or as
      valid scar data? Near-zero bipolar voltage is clinically significant
      (indicates dense scar tissue). This is a patient-safety question that
      must be confirmed with the clinical team.
    - Is fill_threshold=11mm clinically validated? The left ventricle is
      ~50mm across, so 11mm is a large radius.
    - Are voltage values in Volts (requiring ×1000 to get mV) or already
      in mV? The ×1000 multiplication is unverified.
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

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Rotated vertex coordinates.
    pixel_size : float
        In-plane pixel size in mm.
    slice_thickness : float
        Slice thickness in mm.

    Returns
    -------
    VolumeGeometry
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

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Rotated vertex coordinates.
    voltage : np.ndarray, shape (N,)
        Per-vertex voltage values (bipolar or unipolar).
        OPEN QUESTION: units assumed to be Volts; multiplied by 1000 → mV.
    x_car, y_car, z_car : np.ndarray, shape (M,)
        Rotated catheter contact point coordinates.
    geom : VolumeGeometry
        Bounding box / grid info from compute_geometry().
    slice_thickness : float
        Slice thickness in mm.
    fill_threshold : float
        Maximum distance (mm) from a catheter point for a vertex to
        be considered "mapped".

    Returns
    -------
    slice_data : np.ndarray, shape (n_slices, N, 3)
        x, y, z of each point per slice (0 where point not in that slice).
    color_data : np.ndarray, shape (n_slices, N)
        Voltage × 1000 for mapped points, -500 for unmapped.
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