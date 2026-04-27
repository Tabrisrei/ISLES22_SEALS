# Propagation and nnUNet for efficient Acute Ischemic Stroke Segmentation

## Introduction

This is the repository of the Top-1 team SEALS submission for the MICCAI ISLES22 competition.

The repository ships with pre-trained models for the ISLES22 dataset, and now also includes a full **training pipeline** so that you can adapt the method to your own dataset.

---

## Installation

This repository is based on nnUNet. We recommend using a conda environment.

1. Create and activate a conda environment.

```bash
conda create -n nnunet python=3.9
conda activate nnunet
```

2. Install PyTorch with CUDA support.

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

3. Clone this repository.

```bash
cd ~
git clone https://github.com/Tabrisrei/ISLES22_SEALS.git
```

4. Install the package (installs nnUNet in editable mode).

```bash
cd ISLES22_SEALS
pip install -e .
```

5. Install pinned dependency versions to avoid compatibility issues.
   You may need SimpleITK < 2.0.2 to avoid unpredictable errors.

```bash
pip install -r requirements.txt
```

---

## Inference with Pre-trained Models (ISLES22 results reproduction)

Use these steps to reproduce the ISLES22 competition results with the provided models.

1. Download the model archive from [Google Drive](https://drive.google.com/file/d/193l7WTcedo-yvqH4MvIzmIyPpECMdKXQ/view?usp=sharing) and unzip it.
2. Place the `nnUNet_trained_models` folder at `~/ISLES22_SEALS/data/`.
3. Convert your DWI and ADC images to `.mha` format (metadata must be preserved).
4. Place your images in the input directories:
   - DWI: `/input/images/dwi-brain-mri/`
   - ADC: `/input/images/adc-brain-mri/`

   Alternatively, edit `raw_data_dir` in
   `nnunet/dataset_conversion/Task500_Ischemic_Stroke_Test.py`.

5. Run inference.

```bash
bash nnunet_launcher.sh
```

Results are written to `/output/images/stroke-lesion-segmentation/`.

---

## Training on Custom Datasets

Follow these steps to train SEALS on your own dataset and use the resulting
model for inference.

### Prerequisites

- A segmentation dataset in BIDS-like format (see layout below).
- At least one NVIDIA GPU with 12 GB VRAM (24 GB+ recommended for `3d_fullres`).
- Sufficient disk space: approximately 10x the size of your raw data for
  preprocessed files.

### Step 1 — Prepare your dataset

Organize your data in the following BIDS-like structure:

```
data/custom_dataset/
    rawdata/
        sub-case0001/
            ses-0001/
                sub-case0001_ses-0001_dwi.nii.gz
                sub-case0001_ses-0001_adc.nii.gz
    derivatives/
        sub-case0001/
            ses-0001/
                sub-case0001_ses-0001_msk.nii.gz
```

### Step 2 — Create a dataset conversion script

Copy the provided template and adapt it for your dataset.

```bash
cp nnunet/dataset_conversion/TaskXXX_Custom_Dataset_Template.py \
   nnunet/dataset_conversion/Task501_MyDataset.py
```

Open `Task501_MyDataset.py` and edit every line marked with `# TODO`:

| Variable | Description |
|---|---|
| `TASK_ID` | Three-digit integer >= 500 to avoid conflicts with official nnUNet tasks |
| `TASK_NAME` | Must match the directory name you want under `nnUNet_raw_data/` |
| `DATA_ROOT` | Path to your raw dataset root |
| `MODALITY_SUFFIXES` | File-name suffixes for each imaging modality, e.g. `('dwi', 'adc')` |
| `LABELS` | Dictionary mapping integer class indices to class name strings |
| `TEST_RATIO` | Fraction used as held-out test set (0 = all data goes to training) |

### Step 3 — Configure the training launcher

Open `nnunet_training_launcher.sh` and set:

```bash
TASK_ID=501                        # match your conversion script
TASK_NAME="Task501_MyDataset"      # match your conversion script
export CUDA_VISIBLE_DEVICES=0      # GPU(s) to use
```

Choose **one** training mode:
- **Phase 3a** (default, single GPU) — suitable for most setups.
- **Phase 3b** (multi-GPU DDP) — comment out Phase 3a and uncomment Phase 3b.

If you use Phase 3b, also uncomment Phase 4 (`nnUNet_change_trainer_class`) so
that the DDP checkpoints become compatible with single-GPU inference.

### Step 4 — Run training

```bash
bash nnunet_training_launcher.sh
```

The script runs the following stages in sequence:
1. Dataset conversion (raw data -> nnUNet raw-data format).
2. `nnUNet_plan_and_preprocess` — analyses your data and prepares preprocessed files.
3. Training across all 5 cross-validation folds (each fold ~1000 epochs).
4. (DDP only) Trainer-class conversion for inference compatibility.

Training time varies with dataset size and hardware.  A single fold on a
24 GB GPU typically takes 12–24 hours.

### Step 5 — Run inference with your trained model

Edit `nnunet_launcher.sh` and update the two configuration variables at the top:

```bash
TASK_ID=501                    # your training task ID
TRAINER="nnUNetTrainerV2_DDP"  # nnUNetTrainerV2_DDP if trained with DDP
                               # nnUNetTrainerV2    if trained with single GPU
```

Then run inference as usual:

```bash
bash nnunet_launcher.sh
```

### Notes

**DDP vs single-GPU**  
DDP training is faster but requires `nnUNet_change_trainer_class` before
inference (Phase 4 in `nnunet_training_launcher.sh`).  Single-GPU training
(Phase 3a) can be used for inference directly without any conversion step.

**`nnUNet_n_proc_DA`**  
Controls the number of CPU workers for data augmentation.  Set it to roughly
half your physical CPU core count.  The training launcher defaults to 8; the
inference launcher keeps the original value of 24.

**Disk layout after training**  
Trained models are stored at:
```
data/nnUNet_trained_models/nnUNet/3d_fullres/Task501_MyDataset/
    nnUNetTrainerV2_DDP__nnUNetPlansv2.1/   (if DDP)
    nnUNetTrainerV2__nnUNetPlansv2.1/       (if single GPU)
```

---

## Questions

Please contact gtabris@buaa.edu.cn

---

## Acknowledgement

- This code is adapted from [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
- We thank Dr. Fabian Isensee and colleagues for their elegant and efficient code base.
- We thank Dr. Ezequiel and colleagues for their prompt and courteous assistance.
