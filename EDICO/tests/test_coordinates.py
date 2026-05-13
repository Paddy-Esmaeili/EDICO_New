"""
Tests for edico/processing/coordinates.py

Run with:  pytest tests/test_coordinates.py -v
"""

import numpy as np
import pytest

from edico.processing.coordinates import rotate_to_hfs


class TestRotateToHfs:

    def test_known_vertex(self):
        """
        For a single vertex (x=1, y=2, z=3), the rotation should give:
            x_new = -z_old = -3
            y_new =  x_old =  1
            z_new =  y_old =  2
        """
        vertices  = np.array([[1.0, 2.0, 3.0]])
        car_array = np.array([[0.0, 0.0, 0.0]])

        v_rot, _, _, _ = rotate_to_hfs(vertices, car_array)

        np.testing.assert_array_almost_equal(v_rot[0], [-3.0, 1.0, 2.0])

    def test_known_car_point(self):
        """
        For a CAR point (x=1, y=2, z=3), the rotation should give:
            x_car = -z_old = -3
            y_car =  x_old =  1
            z_car =  y_old =  2
        """
        vertices  = np.array([[0.0, 0.0, 0.0]])
        car_array = np.array([[1.0, 2.0, 3.0]])

        _, x_car, y_car, z_car = rotate_to_hfs(vertices, car_array)

        assert x_car[0] == pytest.approx(-3.0)
        assert y_car[0] == pytest.approx(1.0)
        assert z_car[0] == pytest.approx(2.0)

    def test_does_not_modify_input(self):
        """The function must not mutate the input arrays."""
        vertices  = np.array([[1.0, 2.0, 3.0]])
        car_array = np.array([[4.0, 5.0, 6.0]])
        v_copy    = vertices.copy()
        c_copy    = car_array.copy()

        rotate_to_hfs(vertices, car_array)

        np.testing.assert_array_equal(vertices,  v_copy)
        np.testing.assert_array_equal(car_array, c_copy)

    def test_all_zeros(self):
        vertices  = np.zeros((5, 3))
        car_array = np.zeros((3, 3))

        v_rot, x_car, y_car, z_car = rotate_to_hfs(vertices, car_array)

        np.testing.assert_array_equal(v_rot, np.zeros((5, 3)))
        np.testing.assert_array_equal(x_car, np.zeros(3))

    def test_output_shapes(self):
        vertices  = np.random.rand(20, 3)
        car_array = np.random.rand(10, 3)

        v_rot, x_car, y_car, z_car = rotate_to_hfs(vertices, car_array)

        assert v_rot.shape  == (20, 3)
        assert x_car.shape  == (10,)
        assert y_car.shape  == (10,)
        assert z_car.shape  == (10,)

    def test_multiple_points_consistent(self):
        """Each row should be rotated independently and consistently."""
        vertices = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        car_array = np.zeros((1, 3))

        v_rot, _, _, _ = rotate_to_hfs(vertices, car_array)

        np.testing.assert_array_almost_equal(v_rot[0], [0.0,  1.0, 0.0])
        np.testing.assert_array_almost_equal(v_rot[1], [0.0,  0.0, 1.0])
        np.testing.assert_array_almost_equal(v_rot[2], [-1.0, 0.0, 0.0])