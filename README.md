# Property data preprocessing and nnUNet for efficient Acute Ischemic Stroke segmentation

## Introduction

This is the repository of team SEALS submission for ISLES22 competition.

## Installation

This repository is based on nnUNet. Please follow the installation instruction of nnUNet. 
We recommend to install conda environment before installing this repository.
We will provide the installation instruction of this repo.

## Usage

If you do not want to install conda, please skip to the step 3.(We do not recommend to install nnUNet straightly.)

1. Let us hypothesize you are using conda environment named "nnunet", activate conda environment "nnunet"

```bash
conda activate nnunet
```

2. Run the following command to install basic pytorch、torchvision library and corresponding cudatoolkit.

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

3. Clone this repository.

```bash
git clone https://github.com/Tabrisrei/ISLES22_SEALS.git
```

4. Run the following command to install nnUNet.

```bash
cd ISLES22_SEALS
pip install -e .
```

If successful installed all required packages, you can run the following command to test your own dataset.

5. Put pretrain model in the directory "~/ISLES22_SEALS/data/nnUNet_trained_models".

6. Convert your dwi and adc image to mha format.

7. Put your dwi and adc data in `/input/images/dwi-brain-mri/` and `/input/images/adc-brain-mri/` folder respectively.

8. Run the following command to reproduce the results.

```shell
bash nnunet_launcher.sh
```

After the results are generated, you can check the results in `/output/images/stroke-lesion-segmentation/` folder.

## Questions

Please contact gtabris@buaa.edu.cn

## Acknowledgement

- This code is adapted from [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- We thank Dr. Fabian Isensee etc. for their elegant and efficient code base.
