# Property data preprocessing and nnUNet for efficient Acute Ischemic Stroke segmentation

## Introduction

This is the repository of Top1 team SEALS submission for MICCAI ISLES22 competition.

## Installation

This repository is based on nnUNet. Please follow the installation instruction of nnUNet.
We recommend to install conda environment before installing this repository.
We will provide the installation instruction of this repo.

## Usage

If you do not want to install conda, please skip to the step 3.(We do not recommend to install nnUNet straightly.)

1. Let us hypothesis you are using conda environment named "nnunet", activate conda environment "nnunet"

```bash
conda activate nnunet
```

2. Run the following command to install basic pytorch、torchvision library and corresponding cudatoolkit.

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

3. Clone this repository.

```bash
cd ~
git clone https://github.com/Tabrisrei/ISLES22_SEALS.git
```

4. Run the following command to install nnUNet.

```bash
cd ISLES22_SEALS
pip install -e .
```

5. Maybe you can check if successfully installed required package version. (nnunet installation process did not strictly determine the package version, which may raise some unpredictable error.)

```bash
cd ISLES22_SEALS
pip install -r requirements.txt
```


If successfully installed all required packages, you can run the following command to test your own dataset(You may need to install SimpleITK version under 2.0.2 to avoid some unpredictable errors).

5. Download zip file of our models from [Google Drive](https://drive.google.com/file/d/193l7WTcedo-yvqH4MvIzmIyPpECMdKXQ/view?usp=sharing) and unzip it.
6. Put the "nnUNet_trained_models" folder you've got into the directory `~/ISLES22_SEALS/data/`.
7. Convert your dwi and adc image to mha format (Do not forget metadata)
8. Put your dwi and adc data into `/input/images/dwi-brain-mri/` and `/input/images/adc-brain-mri/` folder respectively. Or you may modify the raw_data_diro in `~/ISLES22_SEALS/nnunet/dataset_conversion/Task500_Ischemic_Stroke_Test.py`
9. Run the following command to generate the results.

```shell
bash nnunet_launcher.sh
```

After the results are generated, you can check the results in `/output/images/stroke-lesion-segmentation/` folder.

## Questions

Please contact gtabris@buaa.edu.cn

## Acknowledgement

- This code is adapted from [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- We thank Dr. Fabian Isensee etc. for their elegant and efficient code base.
- We thank Dr. Ezequiel etc. for their prompt and courteous assistance.
