"""
validation/inputs.py
Validates user inputs before the pipeline starts processing.

These checks run up front so the user gets a clear error message.
"""
"""
import os
import re


def validate_inputs(
    mesh_path: str,
    car_path: str,
    output_dir: str,
    patient_id: str,
    patient_name: str,
    series_description: str,
    pixel_size: int,
    slice_thickness: int,
    fill_threshold: int,
):

    errors = []

    # --- File paths ---
    if not mesh_path:
        errors.append("No .mesh file selected.")
    elif not os.path.isfile(mesh_path):
        errors.append(f"Mesh file not found: {mesh_path}")
    elif not mesh_path.lower().endswith('.mesh'):
        errors.append(f"Selected mesh file does not have a .mesh extension: {mesh_path}")

    if not car_path:
        errors.append("No CAR file selected.")
    elif not os.path.isfile(car_path):
        errors.append(f"CAR file not found: {car_path}")

    if not output_dir:
        errors.append("No output directory selected.")

    # --- Patient ID ---
    if not patient_id:
        errors.append("Patient ID is required.")
    else:
        if len(patient_id) > 64:
            errors.append(f"Patient ID exceeds 64 characters: '{patient_id}'")
        if re.search(r'[\\\/\x00-\x1f]', patient_id):
            errors.append(
                f"Patient ID contains illegal characters (backslashes, forward slashes, "
                f"or control characters): '{patient_id}'"
            )

    # --- Patient name ---
    if patient_name and len(patient_name) > 64:
        errors.append(f"Patient name exceeds 64 characters: '{patient_name}'")
    if patient_name and re.search(r'[\\\x00-\x1f]', patient_name):
        errors.append(f"Patient name contains illegal characters: '{patient_name}'")

    # --- Series description ---
    if series_description and len(series_description) > 64:
        errors.append(f"Series description exceeds 64 characters: '{series_description}'")

    # --- Numeric parameters ---
    if pixel_size <= 0:
        errors.append(f"Pixel size must be greater than 0, got: {pixel_size}")
    if slice_thickness <= 0:
        errors.append(f"Slice thickness must be greater than 0, got: {slice_thickness}")
    if fill_threshold <= 0:
        errors.append(f"Fill threshold must be greater than 0, got: {fill_threshold}")

    if errors:
        raise ValueError(
            "Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
        """