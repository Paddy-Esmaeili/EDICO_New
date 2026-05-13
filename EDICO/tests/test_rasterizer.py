"""
Tests for edico/processing/rasterizer.py

Run with:  pytest tests/test_rasterizer.py -v
"""

import numpy as np
import pytest

from edico.processing.rasterizer import rasterize_slice


def _make_inputs(n_points=10, height=20, width=20):
    """Return empty slice_data and color_data arrays."""
    slice_data = np.zeros((n_points, 3))
    color_data = np.zeros(n_points)
    return slice_data, color_data


class TestRasterizeSliceValid:

    def test_empty_slice_is_all_air(self):
        """A slice with no points should be entirely -1000."""
        slice_data, color_data = _make_inputs(n_points=5)
        result, _ = rasterize_slice(slice_data, color_data, 10, 10,
                                    min_x=0.0, min_y=0.0, pixel_size=1.0)
        assert np.all(result == -1000)

    def test_single_mapped_point(self):
        """One mapped point should appear in exactly one pixel."""
        slice_data = np.zeros((1, 3))
        slice_data[0] = [1.5, 1.5, 0.0]   # falls in pixel (1,1) with min=0, size=1
        color_data = np.array([500.0])

        result, _ = rasterize_slice(slice_data, color_data, 5, 5,
                                    min_x=0.0, min_y=0.0, pixel_size=1.0)

        assert result[1, 1] == 500
        # All other pixels should be air
        assert result[0, 0] == -1000

    def test_unmapped_point_gives_minus_500(self):
        """A point labelled -500 (unmapped) should produce a -500 pixel."""
        slice_data = np.zeros((1, 3))
        slice_data[0] = [0.5, 0.5, 0.0]
        color_data = np.array([-500.0])

        result, _ = rasterize_slice(slice_data, color_data, 5, 5,
                                    min_x=0.0, min_y=0.0, pixel_size=1.0)

        assert result[0, 0] == -500

    def test_multiple_points_same_pixel_averaged(self):
        """Multiple mapped points in one pixel should be averaged."""
        slice_data = np.zeros((3, 3))
        slice_data[:, 0] = 0.5   # all in x-pixel 0
        slice_data[:, 1] = 0.5   # all in y-pixel 0
        color_data = np.array([100.0, 200.0, 300.0])

        result, _ = rasterize_slice(slice_data, color_data, 5, 5,
                                    min_x=0.0, min_y=0.0, pixel_size=1.0)

        assert result[0, 0] == pytest.approx(200.0)

    def test_negative_values_excluded_from_average(self):
        """Negative values mixed with positive: only positives averaged."""
        slice_data = np.zeros((3, 3))
        slice_data[:, 0] = 0.5
        slice_data[:, 1] = 0.5
        color_data = np.array([100.0, -500.0, 300.0])

        result, _ = rasterize_slice(slice_data, color_data, 5, 5,
                                    min_x=0.0, min_y=0.0, pixel_size=1.0)

        # Only 100 and 300 should be averaged
        assert result[0, 0] == pytest.approx(200.0)

    def test_output_dtype_is_int16(self):
        slice_data, color_data = _make_inputs()
        result, _ = rasterize_slice(slice_data, color_data, 10, 10,
                                    min_x=0.0, min_y=0.0, pixel_size=1.0)
        assert result.dtype == np.int16

    def test_output_shape(self):
        slice_data, color_data = _make_inputs()
        result, _ = rasterize_slice(slice_data, color_data, 15, 25,
                                    min_x=0.0, min_y=0.0, pixel_size=1.0)
        assert result.shape == (15, 25)

    def test_voxel_stats_populated(self):
        """Stats dict should have entries for mapped voxels."""
        slice_data = np.zeros((1, 3))
        slice_data[0] = [0.5, 0.5, 0.0]
        color_data = np.array([400.0])

        _, stats = rasterize_slice(slice_data, color_data, 5, 5,
                                   min_x=0.0, min_y=0.0, pixel_size=1.0)

        assert len(stats['values'])   == 1
        assert len(stats['n_points']) == 1
        assert stats['values'][0]     == pytest.approx(400.0)


class TestRasterizeSliceEdgeCases:

    def test_all_unmapped(self):
        """Slice where every point is -500 → every pixel is -500 or -1000."""
        slice_data = np.zeros((5, 3))
        for i in range(5):
            slice_data[i] = [float(i) + 0.5, 0.5, 0.0]
        color_data = np.full(5, -500.0)

        result, stats = rasterize_slice(slice_data, color_data, 10, 10,
                                        min_x=0.0, min_y=0.0, pixel_size=1.0)

        assert np.all((result == -500) | (result == -1000))
        assert len(stats['values']) == 0

    def test_point_outside_bounds_raises(self):
        """A point outside the grid should raise ValueError, not silently clip."""
        slice_data = np.zeros((1, 3))
        slice_data[0] = [99.0, 99.0, 0.0]  # far outside a 5×5 grid
        color_data = np.array([100.0])

        with pytest.raises(ValueError, match="outside the grid bounds"):
            rasterize_slice(slice_data, color_data, 5, 5,
                            min_x=0.0, min_y=0.0, pixel_size=1.0)