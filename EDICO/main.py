"""
main.py
Entry point for Edico — Electroanatomic map DIcom COnverter.

This file handles:
    1. GUI setup and argument parsing
    2. Input validation
    3. Calling the pipeline modules in order
    4. Progress reporting

All actual logic lives in the edico/ submodules.
"""

import os
import sys

from gooey import Gooey, GooeyParser

from edico.io.mesh_reader import parse_mesh
from edico.io.car_reader import parse_car
from edico.io.dicom_writer import write_dicom_series
from edico.processing.coordinates import rotate_to_hfs
from edico.processing.slicer import compute_geometry, assign_points_to_slices
from edico.processing.rasterizer import rasterize_slice
#from edico.validation.inputs import validate_inputs
from edico.utils import prepare_directory


@Gooey()
def main():
    program_desc = (
        "Edico is an Electroanatomic map DIcom COnverter meant for compatibility with\n"
        "treatment planning systems in cardiac radioablation."
    )
    parser = GooeyParser(description=program_desc)

    general = parser.add_argument_group('General')
    general.add_argument("mesh",    metavar='Mesh file',      help="Select your .mesh file",  widget="FileChooser")
    general.add_argument("car",     metavar='CAR file',       help="Select your _car file",    widget="FileChooser")
    general.add_argument("results", metavar='Results folder',
                         help="Output directory. Warning: existing /result/ subfolder will be overwritten.",
                         widget="DirChooser")

    patient = parser.add_argument_group('Patient')
    patient.add_argument('-i', '--patientID',
                         metavar='Patient ID', type=str,
                         help='Required. Max 64 chars. No backslashes or control characters.')
    patient.add_argument('-d', '--description',
                         default="test", metavar='Series description', type=str,
                         help='Max 64 chars. No backslashes or control characters.')
    patient.add_argument('-n', '--patientName',
                         default="test_patient", metavar='Patient name', type=str,
                         help='Family name first, separated by ^. Max 64 chars.')

    advanced = parser.add_argument_group('Advanced')
    advanced.add_argument('-t', '--Threshold',
                          default=11, metavar='Fill threshold', type=int,
                          help='Fill threshold in mm. A mesh point within this distance '
                               'of a catheter contact point is considered "mapped".')
    advanced.add_argument('-p', '--PixelSize',
                          default=1, metavar='Pixel size', type=int,
                          help='In-plane pixel size in mm.')
    advanced.add_argument('-s', '--SliceThickness',
                          default=1, metavar='Slice thickness', type=int,
                          help='Slice thickness in mm.')

    args = parser.parse_args(sys.argv[1:])

    output_dir = os.path.join(args.results, "result")

    # --- Validate all inputs before touching anything ---
    """
    validate_inputs(
        mesh_path=args.mesh,
        car_path=args.car,
        output_dir=output_dir,
        patient_id=args.patientID,
        patient_name=args.patientName,
        series_description=args.description,
        pixel_size=args.PixelSize,
        slice_thickness=args.SliceThickness,
        fill_threshold=args.Threshold,
    )
"""
    prepare_directory(output_dir)

    # --- Read inputs ---
    print("Reading mesh file...")
    vertices, triangles, color_data, header_lines = parse_mesh(args.mesh)

    # Currently only bipolar voltage is used.
    # OPEN QUESTION: add unipolar as a GUI dropdown option (original TODO).
    color_bipolar  = color_data[:, 1]

    print("Reading CAR file...")
    car_array = parse_car(args.car)

    # --- Rotate to DICOM HFS orientation ---
    vertices, x_car, y_car, z_car = rotate_to_hfs(vertices, car_array)

    # --- Compute volume geometry ---
    geom = compute_geometry(vertices, args.PixelSize, args.SliceThickness)

    print(f"Volume: {geom.number_of_slices} slices, "
          f"{geom.figure_height} × {geom.figure_width} pixels per slice")

    # --- Assign points to slices ---
    print("Assigning points to slices...")
    slice_data, color_slice_data = assign_points_to_slices(
        vertices=vertices,
        voltage=color_bipolar,
        x_car=x_car,
        y_car=y_car,
        z_car=z_car,
        geom=geom,
        slice_thickness=args.SliceThickness,
        fill_threshold=args.Threshold,
    )

    # --- Rasterize each slice and collect results ---
    print("Rasterizing slices...")
    dicom_slices = []

    for j in range(geom.number_of_slices):
        print(f"Processing slice {j + 1} of {geom.number_of_slices}")
        sys.stdout.flush()

        z_int16, voxel_stats = rasterize_slice(
            slice_data_row=slice_data[j],
            color_data_row=color_slice_data[j],
            figure_height=geom.figure_height,
            figure_width=geom.figure_width,
            min_x=geom.min_x,
            min_y=geom.min_y,
            pixel_size=args.PixelSize,
        )
        dicom_slices.append(z_int16)

    # --- Write DICOM files ---
    print("Writing DICOM files...")
    write_dicom_series(
        slices=dicom_slices,
        output_dir=output_dir,
        patient_id=args.patientID,
        patient_name=args.patientName,
        series_description=args.description,
        pixel_size=args.PixelSize,
        slice_thickness=args.SliceThickness,
        min_x=geom.min_x,
        min_y=geom.min_y,
        min_z=geom.min_z,
    )

    print()
    print(f"Done. DICOM files saved to: {output_dir}")


if __name__ == "__main__":
    main()