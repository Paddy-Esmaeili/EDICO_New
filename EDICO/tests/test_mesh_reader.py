"""
Tests for edico/io/mesh_reader.py

Run with:  pytest tests/test_mesh_reader.py -v
"""

import os
import textwrap
import numpy as np
import pytest

from edico.io.mesh_reader import parse_mesh


# ---------------------------------------------------------------------------
# Helpers to write synthetic .mesh files
# ---------------------------------------------------------------------------

def _write_mesh(tmp_path, n_vertex, n_triangle, vertices, triangles, colors):
    """Write a minimal valid .mesh file."""
    lines = []
    # 8 header lines
    for i in range(8):
        lines.append(f"header line {i}")
    lines.append(f"NumVertex = {n_vertex}")
    lines.append(f"NumTriangle = {n_triangle}")
    lines.append("")  # blank line before first block

    # Vertex block
    for i, v in enumerate(vertices):
        lines.append(f"{i} = {v[0]} {v[1]} {v[2]}")
    lines.append("")

    # Triangle block
    for i, t in enumerate(triangles):
        lines.append(f"{i} = {t[0]} {t[1]} {t[2]}")
    lines.append("")

    # Color block
    for i, c in enumerate(colors):
        lines.append(f"{i} = " + " ".join(str(x) for x in c))

    p = tmp_path / "test.mesh"
    p.write_text("\n".join(lines))
    return str(p)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseMeshValid:

    def test_returns_correct_shapes(self, tmp_path):
        verts = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        tris  = [[0.0, 1.0, 2.0]]
        colors = [[0.5, 1.2], [0.3, 0.8], [0.0, 0.4]]
        path = _write_mesh(tmp_path, 3, 1, verts, tris, colors)

        vertices, triangles, color_data, header = parse_mesh(path)

        assert vertices.shape  == (3, 3)
        assert triangles.shape == (1, 3)
        assert color_data.shape == (3, 2)

    def test_vertex_values_correct(self, tmp_path):
        verts = [[1.5, 2.5, 3.5]]
        tris  = [[0.0, 0.0, 0.0]]
        colors = [[0.1, 0.2]]
        path = _write_mesh(tmp_path, 1, 1, verts, tris, colors)

        vertices, _, _, _ = parse_mesh(path)

        np.testing.assert_array_almost_equal(vertices[0], [1.5, 2.5, 3.5])

    def test_color_columns_accessible(self, tmp_path):
        verts = [[0.0, 0.0, 0.0]]
        tris  = [[0.0, 0.0, 0.0]]
        colors = [[0.9, 1.5, 2.0]]  # unipolar=0.9, bipolar=1.5
        path = _write_mesh(tmp_path, 1, 1, verts, tris, colors)

        _, _, color_data, _ = parse_mesh(path)

        assert color_data[0, 0] == pytest.approx(0.9)  # unipolar
        assert color_data[0, 1] == pytest.approx(1.5)  # bipolar

    def test_header_lines_preserved(self, tmp_path):
        verts  = [[0.0, 0.0, 0.0]]
        tris   = [[0.0, 0.0, 0.0]]
        colors = [[0.1, 0.2]]
        path = _write_mesh(tmp_path, 1, 1, verts, tris, colors)

        _, _, _, header = parse_mesh(path)

        assert len(header) == 8


class TestParseMeshEdgeCases:

    def test_single_vertex(self, tmp_path):
        path = _write_mesh(tmp_path, 1, 1,
                           [[1.0, 2.0, 3.0]],
                           [[0.0, 0.0, 0.0]],
                           [[0.5, 1.0]])
        vertices, _, _, _ = parse_mesh(path)
        assert vertices.shape == (1, 3)

    def test_extra_color_columns_kept(self, tmp_path):
        """Color data may have more than 2 columns — all should be kept."""
        path = _write_mesh(tmp_path, 1, 1,
                           [[0.0, 0.0, 0.0]],
                           [[0.0, 0.0, 0.0]],
                           [[0.1, 0.2, 0.3, 0.4]])
        _, _, color_data, _ = parse_mesh(path)
        assert color_data.shape[1] >= 4


class TestParseMeshErrors:

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_mesh("/nonexistent/path/file.mesh")

    def test_missing_color_column(self, tmp_path):
        """Only one color column → should raise ValueError."""
        verts  = [[0.0, 0.0, 0.0]]
        tris   = [[0.0, 0.0, 0.0]]
        colors = [[0.5]]  # only unipolar, no bipolar
        path = _write_mesh(tmp_path, 1, 1, verts, tris, colors)

        with pytest.raises(ValueError, match="fewer than 2 columns"):
            parse_mesh(path)

    def test_truncated_file(self, tmp_path):
        """File that ends mid-block should raise ValueError."""
        p = tmp_path / "truncated.mesh"
        p.write_text("header\n" * 8 + "NumVertex = 100\nNumTriangle = 50\n")
        with pytest.raises(ValueError):
            parse_mesh(str(p))