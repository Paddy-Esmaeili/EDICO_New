"""
rasterizer.py
Converts one 2D slice (a sparse set of x, y, voltage points) into a
dense pixel grid suitable for DICOM export.

For each pixel cell:
    - If no points fall in the cell          → -1000  ("air")
    - If all points are unmapped (-500)      → -500   ("unmapped region")
    - If valid (positive) points exist       → mean of positive voltages
    - If only negative non-(-500) values     → -500   (treated as unmapped)

The output is a 2D int16 array matching the DICOM pixel representation.
"""

import numpy as np


def rasterize_slice(
    slice_data_row: np.ndarray,
    color_data_row: np.ndarray,
    figure_height: int,
    figure_width: int,
    min_x: float,
    min_y: float,
    pixel_size: float,
) -> tuple:
    """
    Convert one slice's point cloud into a 2D pixel grid.

    Parameters
    ----------
    slice_data_row : np.ndarray, shape (N, 3)
        x, y, z coordinates of all points for this slice.
        Points not belonging to this slice have all-zero coordinates.
    color_data_row : np.ndarray, shape (N,)
        Voltage values for each point (-500 = unmapped, 0 = not in slice).
    figure_height : int
        Number of pixel rows in the output image.
    figure_width : int
        Number of pixel columns in the output image.
    min_x : float
        Physical x-coordinate of the left edge of the grid.
    min_y : float
        Physical y-coordinate of the bottom edge of the grid.
    pixel_size : float
        Size of each pixel in mm.

    Returns
    -------
    z_int16 : np.ndarray, shape (figure_height, figure_width), dtype int16
        Rasterized pixel values.
    voxel_stats : dict
        Diagnostic statistics for this slice:
            'values'    : list of mean voltages for mapped voxels
            'n_points'  : list of point counts per mapped voxel
            'coeff_var' : list of coefficient of variation per mapped voxel
    """
    X = slice_data_row[:, 0]
    Y = slice_data_row[:, 1]
    color = color_data_row

    # Remove points that are not in this slice (stored as 0,0,0 / 0)
    invalid = ((X == 0) & (Y == 0) & (color == 0)) | np.isnan(X) | np.isnan(Y) | np.isnan(color)
    X     = X[~invalid]
    Y     = Y[~invalid]
    color = color[~invalid]

    # Accumulate points into grid cells
    # Z[row][col] is a list of voltage values for that pixel
    Z = [[[] for _ in range(figure_width)] for _ in range(figure_height)]

    for x, y, c in zip(X, Y, color):
        col = int(np.floor((x - min_x) / pixel_size))
        row = int(np.floor((y - min_y) / pixel_size))

        if not (0 <= row < figure_height and 0 <= col < figure_width):
            raise ValueError(
                f"Point ({x:.2f}, {y:.2f}) maps to grid cell ({row}, {col}) "
                f"which is outside the grid bounds "
                f"({figure_height} rows × {figure_width} cols). "
                f"This may indicate a bounding box calculation error."
            )

        Z[row][col].append(c)

    # Aggregate each pixel cell into a single value
    Z_avg = np.zeros((figure_height, figure_width))
    voxel_values, voxel_n_points, voxel_coeff_var = [], [], []

    for row in range(figure_height):
        for col in range(figure_width):
            colors = Z[row][col]

            if not colors:
                # Empty cell → air
                Z_avg[row, col] = -1000

            elif np.mean(colors) == -500:
                # All points in this cell are unmapped
                Z_avg[row, col] = -500

            else:
                # At least some mapped points: average only positive values
                valid = np.array([v for v in colors if v > 0])

                if len(valid) == 0:
                    # Only negative non-(-500) values → treat as unmapped
                    Z_avg[row, col] = -500
                else:
                    mean_v = np.mean(valid)
                    std_v  = np.std(valid, ddof=0)
                    cv     = std_v / mean_v if mean_v != 0 else np.nan

                    Z_avg[row, col] = mean_v

                    voxel_values.append(mean_v)
                    voxel_n_points.append(len(valid))
                    voxel_coeff_var.append(cv)

    z_int16 = np.round(Z_avg).astype(np.int16)

    voxel_stats = {
        'values':    voxel_values,
        'n_points':  voxel_n_points,
        'coeff_var': voxel_coeff_var,
    }

    return z_int16, voxel_stats