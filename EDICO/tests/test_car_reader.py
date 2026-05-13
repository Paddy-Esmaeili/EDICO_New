"""
Tests for edico/io/car_reader.py

Run with:  pytest tests/test_car_reader.py -v
"""

import numpy as np
import pytest

from edico.io.car_reader import parse_car


def _write_car(tmp_path, lines):
    p = tmp_path / "test_car"
    p.write_text("\n".join(lines))
    return str(p)


class TestParseCarValid:

    def test_returns_correct_shape(self, tmp_path):
        lines = [
            "P 0 label tag 1.0 2.0 3.0 extra col",
            "P 1 label tag 4.0 5.0 6.0 extra col",
        ]
        path = _write_car(tmp_path, lines)
        result = parse_car(path)
        assert result.shape == (2, 3)

    def test_correct_values(self, tmp_path):
        lines = ["P 0 label tag 10.5 20.5 30.5 extra"]
        path = _write_car(tmp_path, lines)
        result = parse_car(path)
        np.testing.assert_array_almost_equal(result[0], [10.5, 20.5, 30.5])

    def test_non_p_lines_ignored(self, tmp_path):
        lines = [
            "# comment line",
            "H some header",
            "P 0 label tag 1.0 2.0 3.0",
            "M some other record",
        ]
        path = _write_car(tmp_path, lines)
        result = parse_car(path)
        assert result.shape == (1, 3)

    def test_many_points(self, tmp_path):
        lines = [f"P {i} label tag {float(i)} {float(i+1)} {float(i+2)}" for i in range(100)]
        path = _write_car(tmp_path, lines)
        result = parse_car(path)
        assert result.shape == (100, 3)


class TestParseCarErrors:

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_car("/nonexistent/path/car_file")

    def test_no_p_lines(self, tmp_path):
        lines = ["# only comments", "H header line"]
        path = _write_car(tmp_path, lines)
        with pytest.raises(ValueError, match="No catheter contact points"):
            parse_car(path)

    def test_p_line_too_short(self, tmp_path):
        lines = ["P 0 only_three_cols"]  # fewer than 7 columns
        path = _write_car(tmp_path, lines)
        with pytest.raises(ValueError):
            parse_car(path)