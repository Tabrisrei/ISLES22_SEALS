#!/usr/bin/env bash
# =============================================================================
# nnunet_training_launcher.sh
#
# End-to-end training launcher for ISLES22_SEALS on a custom dataset.
# This script is independent of nnunet_launcher.sh (inference only) and does
# not affect the Docker ENTRYPOINT.
#
# Workflow
# --------
#   Phase 0  Environment variables & user configuration
#   Phase 1  Dataset conversion  (raw data -> nnUNet raw_data format)
#   Phase 2  Preprocessing       (nnUNet_plan_and_preprocess)
#   Phase 3a Training, single GPU (nnUNet_train, 5 folds)
#   Phase 3b Training, multi-GPU DDP (run_training_DDP.py, 5 folds)
#            --> choose Phase 3a OR Phase 3b; comment out the other
#   Phase 4  Trainer-class conversion (DDP -> nnUNetTrainerV2, for inference)
#            --> only needed when Phase 3b was used
#
# Usage
# -----
#   1. Prepare your dataset and fill in Section "User Configuration" below.
#   2. Choose Phase 3a (single GPU) or Phase 3b (multi-GPU DDP).
#   3. Run:  bash nnunet_training_launcher.sh
#
# After training, point nnunet_launcher.sh at your trained model by setting
# TASK_ID and TRAINER in that script to match the values used here.
# =============================================================================


# =============================================================================
# Phase 0 — Environment variables & user configuration
# =============================================================================

export nnUNet_raw_data_base="data/nnUNet_raw_data_base"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export RESULTS_FOLDER="data/nnUNet_trained_models"

# Number of CPU workers for data augmentation.
# A good starting point is half the number of physical CPU cores.
export nnUNet_n_proc_DA=8

# --------------------------------------------------------------------------
# TODO: set these to match your dataset conversion script
# --------------------------------------------------------------------------
TASK_ID=501                           # three-digit task ID (>= 500 recommended)
TASK_NAME="Task501_CustomDataset"     # must match the directory name created
                                      # by your dataset conversion script

# --------------------------------------------------------------------------
# TODO: GPU configuration
#       For Phase 3a (single GPU): set CUDA_VISIBLE_DEVICES to one GPU index.
#       For Phase 3b (multi-GPU) : set CUDA_VISIBLE_DEVICES to a comma-
#                                  separated list, and NPROC to GPU count.
# --------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
NPROC=2                               # number of GPUs for DDP (Phase 3b only)
MASTER_PORT=29532                     # DDP master port; change if in use


# =============================================================================
# Phase 1 — Dataset conversion
# =============================================================================
# Converts your raw dataset into nnUNet raw_data format.
# Edit nnunet/dataset_conversion/TaskXXX_Custom_Dataset_Template.py first,
# then update the path below.

# TODO: replace with the path to your converted dataset script
python nnunet/dataset_conversion/TaskXXX_Custom_Dataset_Template.py


# =============================================================================
# Phase 2 — Preprocessing (plan and preprocess)
# =============================================================================
# Analyses the dataset, creates experiment plans, and generates preprocessed
# data.  This can take tens of minutes to several hours depending on dataset
# size.  Run once; subsequent training runs can skip this phase.

nnUNet_plan_and_preprocess -t $TASK_ID --verify_dataset_integrity


# =============================================================================
# Phase 3a — Single-GPU training  (comment out if using Phase 3b)
# =============================================================================
# Trains all 5 cross-validation folds sequentially on one GPU.
# Each fold runs for 1000 epochs by default (~12-24 h per fold on a modern GPU).

for fold in 0 1 2 3 4; do
    nnUNet_train 3d_fullres nnUNetTrainerV2 $TASK_ID $fold \
        --disable_postprocessing_on_folds
done


# =============================================================================
# Phase 3b — Multi-GPU DDP training  (comment out if using Phase 3a)
# =============================================================================
# Trains all 5 folds sequentially using PyTorch DistributedDataParallel.
# --dbs distributes the batch size across GPUs (matches the original SEALS
# training setup); remove --dbs to give each GPU a full batch instead.
#
# After Phase 3b you MUST also run Phase 4 before running inference.

# for fold in 0 1 2 3 4; do
#     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
#     python -m torch.distributed.launch \
#         --master_port=$MASTER_PORT \
#         --nproc_per_node=$NPROC \
#     nnunet/run/run_training_DDP.py \
#         3d_fullres \
#         nnUNetTrainerV2_DDP \
#         $TASK_ID \
#         $fold \
#         --disable_postprocessing_on_folds \
#         --dbs
# done


# =============================================================================
# Phase 4 — Trainer-class conversion  (DDP only; skip when using Phase 3a)
# =============================================================================
# nnUNet_predict expects the trainer class name stored in the checkpoint pkl
# files to be nnUNetTrainerV2.  After DDP training the stored name is
# nnUNetTrainerV2_DDP, which causes a mismatch.  This command rewrites the
# pkl files so single-GPU inference works correctly.

# nnUNet_change_trainer_class \
#     -i $RESULTS_FOLDER/nnUNet/3d_fullres/${TASK_NAME}/nnUNetTrainerV2_DDP__nnUNetPlansv2.1 \
#     -tr nnUNetTrainerV2
