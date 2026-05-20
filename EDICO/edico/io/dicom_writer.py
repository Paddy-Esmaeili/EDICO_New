"""
dicom_writer.py
Creates individual DICOM CT slice files from 2D pixel arrays.
"""

import os
import datetime
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID


def write_dicom_series(
    slices: list,
    output_dir: str,
    patient_id: str,
    patient_name: str,
    series_description: str,
    pixel_size: float,
    slice_thickness: float,
    min_x: float,
    min_y: float,
    min_z: float,
):
    """
    Write a full series of DICOM slices to disk.

    Parameters
    ----------
    slices : list of np.ndarray (int16)
        One 2D array per slice, in order from bottom (min_z) to top.
    output_dir : str
        Directory where .dcm files will be written.
    patient_id : str
        Patient identifier, used in filenames and DICOM tags.
    patient_name : str
        Patient name for DICOM tags.
    series_description : str
        Series description for DICOM tags.
    pixel_size : float
        In-plane pixel size in mm (same for x and y).
    slice_thickness : float
        Distance between slices in mm.
    min_x, min_y, min_z : float
        Physical coordinates of the lower-left-bottom corner of the volume,
        used to populate ImagePositionPatient for each slice.
    """
    # Generate UIDs shared across the entire series
    sop_uid_stem = pydicom.uid.generate_uid()[:-4]
    series_instance_uid = pydicom.uid.generate_uid(prefix='1.3.6.1.4.1.9590.100.1.4.')
    study_instance_uid  = pydicom.uid.generate_uid(prefix='1.3.6.1.4.1.9590.100.1.4.')
    frame_of_reference_uid = pydicom.uid.generate_uid()

    for index, image_array in enumerate(slices):
        _write_single_slice(
            index=index,
            image_array=image_array,
            output_dir=output_dir,
            patient_id=patient_id,
            patient_name=patient_name,
            series_description=series_description,
            pixel_size=pixel_size,
            slice_thickness=slice_thickness,
            min_x=min_x,
            min_y=min_y,
            min_z=min_z,
            sop_uid_stem=sop_uid_stem,
            series_instance_uid=series_instance_uid,
            study_instance_uid=study_instance_uid,
            frame_of_reference_uid=frame_of_reference_uid,
        )


def _build_file_meta(sop_instance_uid: str) -> FileMetaDataset:
    """Create the DICOM file meta header."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = UID("1.2.840.10008.5.1.4.1.1.2")
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID    = UID("1.3.6.1.4.1.9590.100.1.3.100.7.1")
    file_meta.TransferSyntaxUID         = pydicom.uid.ImplicitVRLittleEndian
    file_meta.is_implicit_VR            = True
    file_meta.is_little_endian          = True
    file_meta.original_character_Set    = ['latin_1']
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    return file_meta


def _write_single_slice(
    index: int,
    image_array: np.ndarray,
    output_dir: str,
    patient_id: str,
    patient_name: str,
    series_description: str,
    pixel_size: float,
    slice_thickness: float,
    min_x: float,
    min_y: float,
    min_z: float,
    sop_uid_stem: str,
    series_instance_uid: str,
    study_instance_uid: str,
    frame_of_reference_uid: str,
):
    """Write one DICOM slice file."""
    filename  = f"patient_{patient_id}_slice_{index:03d}.dcm"
    filepath  = os.path.join(output_dir, filename)

    sop_instance_uid = f"{index + 1}.{sop_uid_stem}"
    file_meta = _build_file_meta(sop_instance_uid)

    ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.set_original_encoding(is_implicit_vr=True, is_little_endian=True, character_encoding='latin-1')

    # --- Identification ---
    ds.SOPClassUID    = UID("1.2.840.10008.5.1.4.1.1.2")
    ds.SOPInstanceUID = sop_instance_uid
    ds.Modality       = 'CT'

    # --- Patient ---
    ds.PatientName      = patient_name
    ds.PatientID        = patient_id
    ds.PatientBirthDate = ""
    ds.PatientSex       = ""

    # --- Study / Series ---
    ds.StudyInstanceUID   = study_instance_uid
    ds.SeriesInstanceUID  = series_instance_uid
    ds.FrameOfReferenceUID = frame_of_reference_uid
    ds.SeriesDescription  = series_description
    ds.InstanceNumber     = index
    ds.StudyDate          = ""
    ds.StudyTime          = ""
    ds.StudyID            = ""
    ds.SeriesNumber       = None
    ds.AcquisitionNumber  = None
    ds.AccessionNumber    = ""

    # --- Acquisition ---
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime("%Y%m%d")
    ds.ContentTime = dt.strftime("%H%M%S.%f")
    ds.ImageType   = ['ORIGINAL', 'PRIMARY']
    ds.Manufacturer = ""
    ds.ReferringPhysicianName = ""
    ds.KVP = None

    # --- Geometry ---
    # OPEN QUESTION: ImageOrientationPatient = [0,1,0,1,0,0] was set
    # as "testing this -- might not be needed" in the original code.
    # The commented-out alternative was [-1,0,0,0,1,0]. This should
    # be validated against the treatment planning system.
    ds.ImageOrientationPatient = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    ds.PatientPosition         = ""
    ds.PatientOrientation      = ""
    ds.PositionReferenceIndicator = ""

    # z-position of this slice in physical space
    z_position = round(min_z + index * slice_thickness, 10)
    ds.ImagePositionPatient = [str(min_y), str(min_x), str(z_position)]
    ds.SliceThickness       = slice_thickness
    ds.PixelSpacing         = [pixel_size, pixel_size]

    # --- Pixel data ---
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel    = 1
    ds.BitsAllocated      = 16
    ds.BitsStored         = 16
    ds.HighBit            = 15
    ds.PixelRepresentation = 1       # signed int16
    ds.RescaleSlope       = 1
    ds.RescaleIntercept   = 0
    ds.RescaleType        = 'HU'
    ds.Units              = 'HU'
    ds.Rows, ds.Columns   = image_array.shape
    ds.PixelData          = image_array.tobytes()

    # Remove optional tags that cause issues with some TPS software
    for tag in [(0x0028, 0x0107), (0x0028, 0x0106)]:
        if tag in ds:
            del ds[tag]

    ds.save_as(filepath, enforce_file_format=True)