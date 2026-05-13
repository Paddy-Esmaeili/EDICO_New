"""
Tests for edico/validation/inputs.py

Run with:  pytest tests/test_validation.py -v
"""

import os
import pytest

from edico.validation.inputs import validate_inputs


def _valid_kwargs(tmp_path):
    """Return a dict of valid inputs for testing."""
    mesh = tmp_path / "test.mesh"
    car  = tmp_path / "test_car"
    mesh.write_text("x")
    car.write_text("x")
    return dict(
        mesh_path=str(mesh),
        car_path=str(car),
        output_dir=str(tmp_path / "output"),
        patient_id="PAT001",
        patient_name="Smith^John",
        series_description="Bipolar map",
        pixel_size=1,
        slice_thickness=1,
        fill_threshold=11,
    )


class TestValidateInputsValid:

    def test_valid_inputs_pass(self, tmp_path):
        """Well-formed inputs should not raise."""
        validate_inputs(**_valid_kwargs(tmp_path))

    def test_minimal_patient_id(self, tmp_path):
        kwargs = _valid_kwargs(tmp_path)
        kwargs['patient_id'] = "X"
        validate_inputs(**kwargs)


class TestValidateInputsErrors:

    def test_missing_mesh(self, tmp_path):
        kwargs = _valid_kwargs(tmp_path)
        kwargs['mesh_path'] = "/does/not/exist.mesh"
        with pytest.raises(ValueError, match="not found"):
            validate_inputs(**kwargs)

    def test_wrong_mesh_extension(self, tmp_path):
        wrong = tmp_path / "file.txt"
        wrong.write_text("x")
        kwargs = _valid_kwargs(tmp_path)
        kwargs['mesh_path'] = str(wrong)
        with pytest.raises(ValueError, match=".mesh extension"):
            validate_inputs(**kwargs)

    def test_missing_car(self, tmp_path):
        kwargs = _valid_kwargs(tmp_path)
        kwargs['car_path'] = "/does/not/exist_car"
        with pytest.raises(ValueError, match="not found"):
            validate_inputs(**kwargs)

    def test_missing_patient_id(self, tmp_path):
        kwargs = _valid_kwargs(tmp_path)
        kwargs['patient_id'] = ""
        with pytest.raises(ValueError, match="Patient ID is required"):
            validate_inputs(**kwargs)

    def test_patient_id_too_long(self, tmp_path):
        kwargs = _valid_kwargs(tmp_path)
        kwargs['patient_id'] = "A" * 65
        with pytest.raises(ValueError, match="64 characters"):
            validate_inputs(**kwargs)

    def test_patient_id_with_backslash(self, tmp_path):
        kwargs = _valid_kwargs(tmp_path)
        kwargs['patient_id'] = "PAT\\001"
        with pytest.raises(ValueError, match="illegal characters"):
            validate_inputs(**kwargs)

    def test_zero_pixel_size(self, tmp_path):
        kwargs = _valid_kwargs(tmp_path)
        kwargs['pixel_size'] = 0
        with pytest.raises(ValueError, match="Pixel size"):
            validate_inputs(**kwargs)

    def test_negative_slice_thickness(self, tmp_path):
        kwargs = _valid_kwargs(tmp_path)
        kwargs['slice_thickness'] = -1
        with pytest.raises(ValueError, match="Slice thickness"):
            validate_inputs(**kwargs)

    def test_zero_fill_threshold(self, tmp_path):
        kwargs = _valid_kwargs(tmp_path)
        kwargs['fill_threshold'] = 0
        with pytest.raises(ValueError, match="Fill threshold"):
            validate_inputs(**kwargs)

    def test_multiple_errors_reported_together(self, tmp_path):
        """All errors should be reported at once, not just the first."""
        kwargs = _valid_kwargs(tmp_path)
        kwargs['patient_id']   = ""
        kwargs['pixel_size']   = 0
        kwargs['slice_thickness'] = -5

        with pytest.raises(ValueError) as exc_info:
            validate_inputs(**kwargs)

        msg = str(exc_info.value)
        assert "Patient ID" in msg
        assert "Pixel size" in msg
        assert "Slice thickness" in msg