"""
rasterizer.py
Converts one 2D slice (a sparse set of x, y, voltage points) into a
dense pixel grid suitable for DICOM export.
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
