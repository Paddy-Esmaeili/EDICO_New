# %% --------------------------------------------------------------------
# %
# % MESH TO DICOM CONVERSION PIPELINE
# % Developed by Sarah Lee (formerly Konermann)
# % Continued by Yuliya Shpunarska
# % Pater Lab, McGill University Health Centre
# % Montreal, Canada
# %
# %% ----------------------------------------------------------------------

import os
import shutil
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
import pydicom.uid as uid
from pydicom.uid import UID
from gooey import Gooey, GooeyParser
import sys
import datetime

def prepare_directory(path):
    """helper function to create a results folder within the study folder. Deletes
    existing sub-folders and files."""
    if os.path.exists(path):
        # Clear all files in the directory
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(path)

@Gooey() # GUI
def main():
    # setup for the GUI
    program_desc = "Edico is an Electroanatomic map DIcom COnverter meant for compatibility with \ntreatment planning systems in cardiac radioablation."
    parser = GooeyParser(description=program_desc)

    general_options = parser.add_argument_group('General')
    general_options.add_argument(
        "mesh", metavar='Mesh file', help="Select your .mesh file", widget="FileChooser")
    general_options.add_argument(
        "car", metavar='CAR file', help="Select your _car file", widget="FileChooser")
    general_options.add_argument(
        "results", metavar='Results folder',
        help="Output directory where Edico will put the DICOM files. Warning: Edico will overwrite any existing /results/ folder inside this directory.",
        widget="DirChooser")

    patient_options = parser.add_argument_group('Patient')
    patient_options.add_argument('-i', '--patientID',
                                metavar='Patient ID',
                                type=str,
                                help='No backslashes or control characters. 64 chars maximum.')
    patient_options.add_argument('-d', '--description',
                                default="test",
                                metavar='Series description',
                                type=str,
                                help='No backslashes or control characters. 64 chars maximum.')
    patient_options.add_argument('-n', '--patientName',
                                default="test_patient",
                                metavar='Patient name',
                                type=str,
                                help='Starting with family name. Separate names with ^. No backslashes or control characters. 64 chars maximum.')


    advanced_options = parser.add_argument_group('Advanced')
    advanced_options.add_argument('-t', '--Threshold',
                                default=11,
                                metavar='Fill threshold',
                                type=int,
                                help='Fill threshold (in mm)')
    advanced_options.add_argument('-p', '--PixelSize',
                                default=1,
                                metavar='Pixel size',
                                type=int,
                                help='Pixel size (in mm)')
    advanced_options.add_argument('-s', '--SliceThickness',
                                default=1,
                                metavar='Slice thickness',
                                type=int,
                                help='Slice thickness (in mm)')

    args = parser.parse_args(sys.argv[1:])

    # files and folders taken from GUI
    file_location_mesh = args.mesh
    file_location_car = args.car
    outputDirectory_fullplot_DCM = os.path.join(args.results, "result")

    # options taken from GUI
    series_description = args.description # optional
    patient_name = args.patientName # optional
    patient_id = args.patientID # used for naming the output files

    slice_thickness = args.SliceThickness  # mm
    pixel_size = args.PixelSize       # mm
    fill_threshold = args.Threshold  # mm

    # --- CREATE OR CLEAN DIRECTORIES ---
    # TODO: reorganize the code so it's not.. whatever this is

    prepare_directory(outputDirectory_fullplot_DCM)

    def create_meta():
        # Populate required values for file meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = UID("1.3.6.1.4.1.9590.100.1.3.100.7.1")
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        file_meta.is_implicit_VR = True
        file_meta.is_little_endian = True
        file_meta.original_character_Set = ['latin_1']
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        return file_meta

    def create_dicom_slice(index, image_array, sop_uid_identical_part, seriesInstanceUID, studyInstanceUID, new_frame_of_reference_uid, output_dir=outputDirectory_fullplot_DCM):
        '''
        Create a .dcm file containing one slice.

        Parameters:
            index: (int) index of the slice in the end 3D volume
            image_array: (2D array) data to turn into grayscale pixel values
            output_dir: (str) Optional. Directory where to put the .dcm file. By default, this is outputDirectory_fullplot_DCM specified at the beginning.
        Returns:
            Nothing. Saves the .dcm file in the temp output folder.
        '''
        filename = f"patient_{patient_id}_slice_{index:03d}.dcm"
        filepath = os.path.join(output_dir, filename)

        # Set the orientation of the patient
        image_orientation_patient = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0] # testing this -- might not be needed
        #image_orientation_patient = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        # Create meta info
        new_sop_instance_uid = f"{index+1}.{sop_uid_identical_part}"

        #file_meta = pydicom.Dataset()
        #file_meta.MediaStorageSOPClassUID = pydicom.uid.UID("1.2.840.10008.5.1.4.1.1.2") # for CT
        #file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID # I dont know what this does
        #file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        # TRYING TO MAKE IT WORK
        #file_meta.MediaStorageSOPInstanceUID = pydicom.uid.UID("1.3.6.1.4.1.9590.100.1.4.86845428203749527321656479100312168227") # matching sarah
        # this is overwritten by enforce_file_format=True
        # TODO: try activating below lines, see if it helps
        #file_meta.ImplementationClassUID = pydicom.uid.UID("1.3.6.1.4.1.9590.100.1.3.100.9.4")
        #file_meta.ImplementationVersionName = "MATLAB IPT 9.4"

        file_meta = create_meta() # NEW: ADDED FROM DAN's CODE

        # Create dataset
        ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.set_original_encoding(is_implicit_vr = True, is_little_endian = True, character_encoding = 'latin-1')

        # TRYING TO MAKE IT work
        ds.ImageType = ['ORIGINAL', 'PRIMARY']
        ds.SOPClassUID = pydicom.uid.UID("1.2.840.10008.5.1.4.1.1.2")
        ds.StudyDate = ""
        ds.StudyTime = ""
        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime("%Y%m%d")
        ds.ContentTime = dt.strftime("%H%M%S.%f")
        ds.AccessionNumber = ""
        ds.Manufacturer = ""
        ds.ReferringPhysicianName = ""
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.KVP = None
        ds.PatientPosition = ""
        ds.StudyID = ""
        ds.SeriesNumber = None
        ds.AcquisitionNumber = None
        ds.PatientOrientation = ""
        ds.PositionReferenceIndicator = ""


        ds.SOPInstanceUID = new_sop_instance_uid # this is the same as sarah's
        ds.FrameOfReferenceUID = new_frame_of_reference_uid # this is the same as sarah's

        # these are needed to coordinate all the frames together
        ds.SeriesInstanceUID = seriesInstanceUID
        ds.StudyInstanceUID = studyInstanceUID

        ds.Modality = 'CT'
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.SeriesDescription = series_description
        ds.InstanceNumber = index

        ds.ImageOrientationPatient = image_orientation_patient
        ds.RescaleType = 'HU'
        ds.Units = 'HU'

        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 16 # breaks completely without this line
        ds.HighBit = 15
        ds.PixelSpacing = [pixel_size, pixel_size]
        ds.Rows, ds.Columns = image_array.shape

        imagePositionPatient = [min_y, min_x, round(min_z + j*slice_thickness, 10)]
        # note: this rounding is needed to fit within the size requirement of DS (decimal string) of the dicom tag
        # the number of decimal points here is arbitrary -- chosen to fit the requirement in most cases (18 characters including - and .)
        ds.ImagePositionPatient = list(map(str, imagePositionPatient))  # Define the image position
        ds.SliceThickness = slice_thickness
        # saving the fill threshold in mm in the (0018,0090) DataCollectionDiameter attribute
        #ds.DataCollectionDiameter = fill_threshold
        ds.RescaleSlope = 1
        ds.RescaleIntercept = 0 # CT rescale intercept
        ds.PixelRepresentation = 1

        #ds.PixelData = image_array.astype(np.uint16).tobytes()
        ds.PixelData = image_array.tobytes()

        # Check if "Largest Pixel Value" (Tag: (0028, 0107)) exists in the DICOM file
        if (0x0028, 0x0107) in ds:
            del ds[0x0028, 0x0107]  # Delete the "Largest Pixel Value" attribute

        # Check if "Smallest Pixel Value" (Tag: (0028, 0106)) exists in the DICOM file
        if (0x0028, 0x0106) in ds:
            del ds[0x0028, 0x0106]  # Delete the "Smallest Pixel Value" attribute

        ds.save_as(filepath, enforce_file_format=True)

    # --- READ MESH FILE ---

    with open(file_location_mesh, 'r') as fid:
        header_lines = [fid.readline().strip() for _ in range(8)]

        n_vertex = int(fid.readline().split('=')[1].strip())
        n_triangle = int(fid.readline().split('=')[1].strip())

        while True: # skips lines to the next block of data
            line = fid.readline().strip()
            if line.startswith("0 ="): break

        # Read vertices
        vertices = []
        for _ in range(n_vertex):

            numbers_only = line.strip().split("=")[1] # keep everything after the "=" sign
            values = list(map(float, numbers_only.strip().split()))
            # this line removes extra whitespace around the line (strip), then splits
            # the columns (split), then converts the numbers in these columns from
            # strings to floats (map), and finally collects the numbers into a list

            line = fid.readline()
            vertices.append(values[:3])  # Only keep x, y, z

        vertices = np.array(vertices)

        while True: # skips lines to the next block of data
            line = fid.readline().strip()
            if line.startswith("0 ="): break

        # Read triangles -- WHY DO WE EVEN NEED  THIS AT ALL?
        triangles = []
        for _ in range(n_triangle):
            numbers_only = line.strip().split("=")[1] # keep everything after the "=" sign
            indices = list(map(float, numbers_only.strip().split()))
            triangles.append(indices[:3])  # Triangle connectivity
            line = fid.readline()
        triangles = np.array(triangles)

        while True: # skips lines to the next block of data
            line = fid.readline().strip()
            if line.startswith("0 ="): break

        # read color data
        ColorData = []
        for _ in range(n_vertex):
            numbers_only = line.strip().split("=")[1] # keep everything after the "=" sign
            indices = list(map(float, numbers_only.strip().split()))
            ColorData.append(indices)  # keep full color data just in case we need it in future versions of Edico
            line = fid.readline()
        ColorData = np.array(ColorData)
        color_Bipolar = ColorData[:,1]
        color_Unipolar = ColorData[:,0]

        # this script currently only uses the bipolar maps, but also works with unipolar if you want to visualize it as a dicom
        metric_for_plotting = color_Bipolar
        # TODO: add this as an option in a drop-down menu for the GUI

    # --- READ CAR FILE ---

    with open(file_location_car, 'r') as f:
        car_values = []

        for line in f:
             if line.startswith("P"):
                 values = list(map(float, line.strip().split()[4:7])) # use 5th, 6th, and 7th column
                 car_values.append(values)

        car_array = np.array(car_values)

    # Perform rotations so images are exported in head-first-supine position
    x_car = -car_array[:,2]
    y_car = car_array[:,0]
    z_car = car_array[:,1]

    temp1 = np.copy(vertices[:,0])
    temp2 = np.copy(vertices[:,1])
    temp3 = np.copy(vertices[:,2])

    vertices[:,0] = -temp3
    vertices[:,1] = temp1
    vertices[:,2] = temp2

    number_of_points = len(vertices[:,1]) # Number of points to be included in images


    # SETTING LIMITS FOR PLOTS WITH 5MM PADDING

    min_x = min(vertices[:,0])-5
    min_y = min(vertices[:,1])-5
    min_z = min(vertices[:,2])-5

    max_x = max(vertices[:,0])+5
    max_y = max(vertices[:,1])+5
    max_z = max(vertices[:,2])+5

    # CREATE SLICES

    number_of_slices = (max_z-min_z)/slice_thickness
    number_of_slices = int(np.floor(number_of_slices)+1) # One added to include ALL data points

    slice_data=np.zeros((number_of_slices, number_of_points, 3)) # For storing point spatial information, 3 corresponds to x, y, z
    color_data=np.zeros((number_of_slices, number_of_points, 1)) # For storing point color information

    slice_start_position = min_z;

    # Initialize arrays
    x_original, y_original, z_original, color_original = [], [], [], []

    # loop through all the slices to collect x, y, z, and color data
    for slice_index in range(number_of_slices):
        # loop through all the points
        # TODO: optimize this with array stuff. Im sure this can be done without this loop to make it run a bit faster
        for i in range(number_of_points):
            z_val = vertices[i, 2]
            # checking if the current z position is in the right slice
            if slice_start_position <= z_val < (slice_start_position + slice_thickness):

                # Assign coordinates
                x_val = vertices[i, 0]
                y_val = vertices[i, 1]

                slice_data[slice_index, i, 0] = x_val
                slice_data[slice_index, i, 1] = y_val
                slice_data[slice_index, i, 2] = z_val

                x_original.append(x_val)
                y_original.append(y_val)
                z_original.append(z_val)

                # Check if any point is within [fill threshold] of any point with
                # coordinates given by x_car, y_car, z_car
                distances = np.sqrt((x_val - x_car)**2 + (y_val - y_car)**2 + (z_val - z_car)**2)
                is_within_range = np.any(distances <= fill_threshold)

                # Assign color based on the metric chosen (usually bipolar map)
                # Also check whether point is valid (greater than zero)
                if is_within_range and metric_for_plotting[i] > 0:
                    color_data[slice_index, i, 0] = metric_for_plotting[i] * 1000
                    color_original.append(color_data[slice_index, i, 0])
                else: # Assign -500 to unmapped regions
                    color_data[slice_index, i, 0] = -500
                    color_original.append(-500)

        # Increment slice position
        slice_start_position += slice_thickness


    figureWidth = int(np.floor((max_x-min_x)/pixel_size)+1) # in pixels
    figureHeight = int(np.floor((max_y-min_y)/pixel_size)+1) # in pixels

    # initialize the coefficient of variation as NaN
    coeffVarInVoxel = np.full((int(figureHeight * figureWidth * number_of_slices), 1), np.nan)


    # initialize empty lists for bookkeeping
    voxelValue, numPointsInVoxel, coeffVarInVoxel = [], [], []

    # Generate a new SOP Instance UID (UUID-based) and trim
    # identical for each patient
    sop_uid_identical_part = uid.generate_uid()
    sop_uid_identical_part = sop_uid_identical_part[:-4]

    # keeps track of the series and the study and pretends it was made by MATLAB
    seriesInstanceUID = pydicom.uid.generate_uid(prefix='1.3.6.1.4.1.9590.100.1.4.')
    studyInstanceUID = pydicom.uid.generate_uid(prefix='1.3.6.1.4.1.9590.100.1.4.')
    # frame of reference
    new_frame_of_reference_uid = pydicom.uid.generate_uid()

    # Loop through slices, creating a dicom image for each
    for j in range(number_of_slices):

        # Create the progress message
        progress_msg = f"Processing slice {j + 1} of {number_of_slices}"
        # Print and overwrite the same line
        #print(f"\r{progress_msg}", end='', flush=True)
        # Print normally
        print(f"{progress_msg}")
        sys.stdout.flush()

        X = slice_data[j, :, 0]
        Y = slice_data[j, :, 1]
        Z_exactvalues = slice_data[j, :, 2]
        color = color_data[j, :, 0]

        # Find indices that contain invalid or (0, 0, 0) points
        indices_to_nan = ((X == 0) & (Y == 0) & (color == 0)) | np.isnan(X) | np.isnan(Y) | np.isnan(color)

        # Create new arrays without NaN values for interpolation
        X = X[~indices_to_nan]
        Y = Y[~indices_to_nan]
        # Z_exactvalues = Z_exactvalues[~indices_to_nan]
        #TODO: is it needed to define X, Y, Z separately here? Can we just use slice_data throughout?
        color = color[~indices_to_nan]

        # Calculate grid cell sizes
        dx = pixel_size
        dy = pixel_size

        # Initialize Z as a list of lists to store multiple colors per grid cell
        Z = [[[] for _ in range(figureWidth)] for _ in range(figureHeight)] # this might break

        # Loop through the data and accumulate colors within the corresponding grid cell
        for x, y, c in zip(X, Y, color):
            # Find the grid cell (row and column) containing the data point
            col = int(np.floor((x - min_x) / dx))
            row = int(np.floor((y - min_y) / dy))

            assert 0 <= row < figureHeight and 0 <= col < figureWidth, "Row or column number calculated is outside of figure bounds"

            Z[row][col].append(c)

        # Initialize Z_avg
        Z_avg = np.zeros((figureHeight, figureWidth))

        # Initialize voxelIndex (assuming starts from 0) -- this is different from matlab and might break
        voxelIndex = 0

        # Calculate the average color for each pixel
        for row in range(figureHeight):
            for col in range(figureWidth):
                colors = Z[row][col]

                if colors and np.mean(colors) == -500: # if it's not empty and is unmapped
                    # Assign -500 to unmapped voxels
                    Z_avg[row, col] = np.mean(colors)

                elif colors: # if it's not empty
                    # Exclude negative values when calculating averages
                    valid_colors = np.array([val for val in colors if val > 0])

                    if len(valid_colors) > 0:

                        # Calculate the mean of the colors
                        mean_colors = np.mean(valid_colors)

                        # Calculate the population standard deviation
                        population_std = np.std(valid_colors, ddof=0)

                        # Calculate the coefficient of variation (CV)
                        coefficient_variation = population_std / mean_colors

                        # For mapped voxels, find average voltage.
                        Z_avg[row, col] = mean_colors
                        #print(valid_colors)
                        #print(mean_colors)


                        # Can be used for debugging
                        voxelValue.append(Z_avg[row, col])
                        numPointsInVoxel.append(len(valid_colors))
                        coeffVarInVoxel.append(coefficient_variation)
                        # TODO: make a class and have this as instance variable?
                    else:
                        # if only negative colors, consider them unmapped
                        Z_avg[row, col] = -500
                        # the original Edico doesn't define what to do if all the colors are negative
                else: # if it's empty
                    # Assign -1000 to "air"
                    Z_avg[row, col] = -1000

                voxelIndex += 1

        # Convert to int16
        Z_avg = np.round(Z_avg) # added this to better match the matlab implementation
        Z_int16 = Z_avg.astype(np.int16)

        ## Uncomment the lines below to plot one slice
        # if j == 57: # plot one slice to inspect the data
        #     import matplotlib.pyplot as plt # for testing only
        #     print(np.shape(Z_int16))
        #     plt.imshow(Z_int16)
        #     plt.show()



        # Make the dicom file
        create_dicom_slice(j, Z_int16, sop_uid_identical_part, seriesInstanceUID, studyInstanceUID, new_frame_of_reference_uid)

    print()
    print(f'DICOM data saved to: {outputDirectory_fullplot_DCM}')

if __name__ == "__main__":
    main()
