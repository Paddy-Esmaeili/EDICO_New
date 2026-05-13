"""
mesh_reader.py
Reads the .mesh file exported by the electroanatomic mapping system (e.g. CARTO).

The file format has:
  - 8 header lines (currently unused but preserved for future use)
  - n_vertex / n_triangle counts
  - Three data blocks separated by blank/comment lines:
      1. Vertex coordinates  (x, y, z, ...)
      2. Triangle indices    (i, j, k, ...)
      3. Color/voltage data  (unipolar, bipolar, ...)

Each data row is prefixed with its 0-based index, e.g.:  0 = 1.23 4.56 7.89
"""

import numpy as np


def parse_mesh(filepath: str):
    """
    Parse a .mesh file.

    Parameters
    ----------
    filepath : str
        Path to the .mesh file.

    Returns
    -------
    vertices : np.ndarray, shape (N, 3)
        x, y, z coordinates of each vertex in the original coordinate system.
    triangles : np.ndarray, shape (M, 3)
        Triangle connectivity as vertex indices.
        NOTE: These are read but not currently used in the pipeline.
              Preserved for future surface-fill implementations.
    color_data : np.ndarray, shape (N, K)
        Per-vertex scalar values.
        Column 0 = unipolar voltage
        Column 1 = bipolar voltage
        Additional columns may exist; their meaning is undocumented.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is malformed or cannot be parsed.
    """
    with open(filepath, 'r') as fid:
        # Read and preserve the 8-line header for future reference.
        # OPEN QUESTION: the header may contain the coordinate system
        # declaration. If so, the rotation in coordinates.py may need
        # to be conditional on this value rather than hardcoded.
        header_lines = [fid.readline().strip() for _ in range(8)]

        try:
            n_vertex = int(fid.readline().split('=')[1].strip())
            n_triangle = int(fid.readline().split('=')[1].strip())
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Could not read vertex/triangle counts from {filepath}. "
                f"File may be malformed or from an unsupported software version."
            ) from e

        # --- Vertices ---
        line = _skip_to_block(fid, filepath, block_name="vertices")
        vertices = _read_data_block(fid, line, n_vertex, cols=3, block_name="vertices")

        # --- Triangles ---
        line = _skip_to_block(fid, filepath, block_name="triangles")
        triangles = _read_data_block(fid, line, n_triangle, cols=3, block_name="triangles")

        # --- Color / voltage data ---
        line = _skip_to_block(fid, filepath, block_name="color data")
        color_data = _read_data_block(fid, line, n_vertex, cols=None, block_name="color data")

    if color_data.ndim == 1 or color_data.shape[1] < 2:
        raise ValueError(
            f"Color data in {filepath} has fewer than 2 columns. "
            f"Expected at least [unipolar, bipolar]."
        )

    return vertices, triangles, color_data, header_lines


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _skip_to_block(fid, filepath: str, block_name: str) -> str:
    """
    Advance the file cursor until a line starting with '0 =' is found.
    Returns that line so the caller can immediately start reading data.

    Raises ValueError if EOF is reached without finding the block.
    """
    for line in fid:
        line = line.strip()
        if line.startswith("0 ="):
            return line
    raise ValueError(
        f"Reached end of file while looking for {block_name} block in {filepath}. "
        f"File may be truncated or from an unsupported software version."
    )


def _read_data_block(fid, first_line: str, n_rows: int, cols, block_name: str) -> np.ndarray:
    """
    Read n_rows rows from the current file position.
    Each row has the format:  <index> = <v1> <v2> ... <vN>

    Parameters
    ----------
    cols : int or None
        If int, only keep the first `cols` values per row.
        If None, keep all values (used for color data with unknown column count).
    """
    data = []
    line = first_line
    for row_num in range(n_rows):
        try:
            numbers_only = line.strip().split("=")[1]
            values = list(map(float, numbers_only.strip().split()))
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Could not parse row {row_num} of {block_name} block: {repr(line)}"
            ) from e

        data.append(values if cols is None else values[:cols])
        line = fid.readline()

    return np.array(data)