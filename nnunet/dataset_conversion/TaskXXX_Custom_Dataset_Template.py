#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Template script for converting a custom dataset to nnUNet format.

Usage
-----
1. Copy this file and rename it to match your task, e.g.:
       cp TaskXXX_Custom_Dataset_Template.py Task501_MyStrokeDataset.py

2. Search for every line marked with  # TODO  and fill in the correct values
   for your dataset.

3. Run the script once to populate the nnUNet raw-data directory:
       python nnunet/dataset_conversion/Task501_MyStrokeDataset.py

Expected raw dataset layout (BIDS-like)
----------------------------------------
<DATA_ROOT>/
    rawdata/
        sub-case0001/
            ses-0001/
                sub-case0001_ses-0001_dwi.nii.gz
                sub-case0001_ses-0001_adc.nii.gz
                ...
    derivatives/
        sub-case0001/
            ses-0001/
                sub-case0001_ses-0001_msk.nii.gz

After running this script the nnUNet raw-data directory will look like:
<nnUNet_raw_data_base>/nnUNet_raw_data/TaskXXX_<name>/
    imagesTr/   <patient_id>_0000.nii.gz  (modality 0)
                <patient_id>_0001.nii.gz  (modality 1)
                ...
    labelsTr/   <patient_id>.nii.gz
    imagesTs/   (empty or filled if you have a held-out test set)
    labelsTs/   (empty or filled if you have a held-out test set)
    dataset.json
"""

import os
from glob import glob

import numpy as np
import SimpleITK as sitk

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data


# ---------------------------------------------------------------------------
# TODO: Set the full task name.
#       Must be "Task<ID>_<YourName>", e.g. "Task501_MyStrokeDataset".
#       This name is used as the directory name under nnUNet_raw_data and must
#       match the TASK_ID / TASK_NAME variables in nnunet_training_launcher.sh.
# ---------------------------------------------------------------------------
TASK_NAME = "Task501_CustomDataset"   # TODO

# ---------------------------------------------------------------------------
# TODO: Set the root directory of your raw dataset
# ---------------------------------------------------------------------------
DATA_ROOT = "data/custom_dataset/"

# ---------------------------------------------------------------------------
# TODO: Define modality suffixes as they appear in your file names.
#       The order here must match the channel order you want nnUNet to see.
#       Example for DWI + ADC:  ('dwi', 'adc')
# ---------------------------------------------------------------------------
MODALITY_SUFFIXES = ('dwi', 'adc')   # TODO

# ---------------------------------------------------------------------------
# TODO: Define your segmentation labels.
#       Key = integer class index, Value = class name string.
# ---------------------------------------------------------------------------
LABELS = {0: 'background', 1: 'lesion'}   # TODO

# ---------------------------------------------------------------------------
# TODO: Set the test split ratio.
#       Every TEST_RATIO-th sample is used as test; 0 means no held-out test.
# ---------------------------------------------------------------------------
TEST_RATIO = 0   # TODO  (0 = all data goes to training)

# ---------------------------------------------------------------------------
# TODO: Set the target voxel spacing for resampling (in mm, x/y/z order).
#       Use None to keep original spacing.
# ---------------------------------------------------------------------------
TARGET_SPACING = [1.0, 1.0, 1.0]   # TODO


# ===========================================================================
# Data loading helper
# ===========================================================================

class CustomDataset:
    """
    Loads file paths from a BIDS-like directory structure.
    Adapt the glob patterns below if your naming convention differs.
    """

    def __init__(self, root: str):
        self.root = root
        self.data_dict: dict = {}

    def load_data(self):
        mask_folder = os.path.join(self.root, 'derivatives')
        raw_folder  = os.path.join(self.root, 'rawdata')

        for key in ['mask_list'] + [f'{s}_list' for s in MODALITY_SUFFIXES]:
            self.data_dict[key] = []

        patient_dirs = sorted(os.listdir(mask_folder))
        for patient in patient_dirs:
            if not patient.startswith('sub'):
                continue

            mask_dir = os.path.join(mask_folder, patient, 'ses-0001')
            raw_dir  = os.path.join(raw_folder,  patient, 'ses-0001')

            masks = glob(os.path.join(mask_dir, '*_msk.nii.gz'))
            if not masks:
                continue
            self.data_dict['mask_list'].append(masks[0])

            for suffix in MODALITY_SUFFIXES:
                # TODO: adjust the glob pattern if your filenames differ
                matches = glob(os.path.join(raw_dir, f'*_{suffix}.nii.gz'))
                assert matches, (
                    f"No file matching '*_{suffix}.nii.gz' found in {raw_dir}. "
                    f"Check MODALITY_SUFFIXES or your file naming convention."
                )
                self.data_dict[f'{suffix}_list'].append(matches[0])

        self.length = len(self.data_dict['mask_list'])
        print(f"Loaded {self.length} cases from {self.root}")


# ===========================================================================
# Resampling utilities  (same logic as Task500_Ischemic_Stroke_Test.py)
# ===========================================================================

def respacing_file(image_file, target_spacing, resample_method):
    """Resample image_file to target_spacing."""
    if not isinstance(image_file, sitk.SimpleITK.Image):
        image_file = sitk.ReadImage(image_file)
    if not isinstance(target_spacing, np.ndarray):
        target_spacing = np.array(target_spacing)

    resampler = sitk.ResampleImageFilter()
    origin_spacing = np.array(image_file.GetSpacing())
    origin_size    = np.array(image_file.GetSize())
    factor         = target_spacing / origin_spacing
    # np.int64 avoids the overflow that np.uint8 causes for images > 255 px
    target_size    = (origin_size / factor).astype(np.int64)

    resampler.SetReferenceImage(image_file)
    resampler.SetOutputSpacing(target_spacing.tolist())
    resampler.SetSize(target_size.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_method)
    return resampler.Execute(image_file)


def reimplement_resize(image_file, target_file,
                       resample_method=sitk.sitkLinear):
    """Resample image_file to match the geometry of target_file."""
    if isinstance(image_file, str):
        image_file = sitk.ReadImage(image_file)
    if isinstance(target_file, str):
        target_file = sitk.ReadImage(target_file)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image_file)
    resampler.SetOutputOrigin(target_file.GetOrigin())
    resampler.SetOutputDirection(target_file.GetDirection())
    resampler.SetOutputSpacing(target_file.GetSpacing())
    resampler.SetSize(target_file.GetSize())
    if resample_method == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt8)
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_method)
    return resampler.Execute(image_file)


# ===========================================================================
# Train / test split
# ===========================================================================

def split_dataset(data_num: int, test_ratio: int):
    """Return (train_indices, test_indices).  test_ratio=0 puts all in train."""
    if test_ratio == 0:
        return list(range(data_num)), []
    train, test = [], []
    for i in range(data_num):
        (test if i % test_ratio == 0 else train).append(i)
    return train, test


# ===========================================================================
# Main conversion
# ===========================================================================

if __name__ == "__main__":
    task_full_name = TASK_NAME

    dataset = CustomDataset(DATA_ROOT)
    dataset.load_data()

    target_base     = join(nnUNet_raw_data, task_full_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    for d in [target_base, target_imagesTr, target_labelsTr,
              target_imagesTs, target_labelsTs]:
        maybe_mkdir_p(d)

    train_idx, test_idx = split_dataset(dataset.length, TEST_RATIO)

    def process_case(idx, out_images_dir, out_labels_dir):
        case_id = str(idx + 1).rjust(4, '0')
        prefix  = f"case_{case_id}"   # TODO: change prefix if desired

        mask_path = dataset.data_dict['mask_list'][idx]

        # Resample each modality to match the mask geometry (preserves labels)
        mask_img = reimplement_resize(
            mask_path, mask_path,
            resample_method=sitk.sitkNearestNeighbor
        )

        for ch_idx, suffix in enumerate(MODALITY_SUFFIXES):
            img_path = dataset.data_dict[f'{suffix}_list'][idx]
            # TODO: switch to respacing_file(img_path, TARGET_SPACING, ...)
            #       if you want a fixed voxel spacing instead of matching mask
            img = reimplement_resize(img_path, mask_path,
                                     resample_method=sitk.sitkLinear)
            sitk.WriteImage(
                img,
                join(out_images_dir, f"{prefix}_{str(ch_idx).rjust(4, '0')}.nii.gz")
            )

        sitk.WriteImage(mask_img, join(out_labels_dir, f"{prefix}.nii.gz"))

    print("Converting training cases ...")
    for idx in train_idx:
        process_case(idx, target_imagesTr, target_labelsTr)

    print("Converting test cases ...")
    for idx in test_idx:
        process_case(idx, target_imagesTs, target_labelsTs)

    generate_dataset_json(
        output_file=join(target_base, 'dataset.json'),
        imagesTr_dir=target_imagesTr,
        imagesTs_dir=target_imagesTs,
        modalities=tuple(MODALITY_SUFFIXES),
        labels=LABELS,
        dataset_name=task_full_name,
        sort_keys=True,
        license="Dataset license"   # TODO: set your dataset license
    )

    print(f"Done. Raw data written to: {target_base}")
