# =============================================================================
# nnunet_launcher.sh  —  Inference pipeline for ISLES22_SEALS
#
# This script is the Docker ENTRYPOINT (see Dockerfile).  It converts a
# single test case, runs 5-fold inference, and writes the final segmentation
# to /output/images/stroke-lesion-segmentation/.
#
# To use with a custom trained model
# ------------------------------------
# 1. Set TASK_ID to the numeric task ID used during training.
# 2. Set TRAINER to the trainer name that appears in the model folder name:
#      - "nnUNetTrainerV2_DDP"  if trained with DDP (Phase 3b)
#      - "nnUNetTrainerV2"      if trained with single GPU (Phase 3a)
# 3. Set MODEL_DIR to the full path of your trained model folder.
#    The nnUNet_change_trainer_class step rewrites the trainer name stored
#    inside the checkpoint pkl files from DDP to the single-GPU variant so
#    that nnUNet_predict can load them.  Skip this step (comment it out) if
#    you trained with single GPU or already ran it from nnunet_training_launcher.sh.
#
# See nnunet_training_launcher.sh for the full training workflow.
# =============================================================================

export nnUNet_raw_data_base="data/nnUNet_raw_data_base"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export RESULTS_FOLDER="data/nnUNet_trained_models"
export nnUNet_n_proc_DA=24

# --------------------------------------------------------------------------
# User configuration
# Defaults reproduce the original ISLES22 SEALS submission results.
# When switching to a custom trained model update all three variables.
# --------------------------------------------------------------------------
TASK_ID=12
TRAINER="nnUNetTrainerV2_DDP"
MODEL_DIR="data/nnUNet_trained_models/nnUNet/3d_fullres/Task012_Ischemic_Stroke_TM_Fullset/nnUNetTrainerV2_DDP__nnUNetPlansv2.1"
# For a custom model trained with DDP uncomment and adjust the line below:
# MODEL_DIR="$RESULTS_FOLDER/nnUNet/3d_fullres/Task501_YourDataset/nnUNetTrainerV2_DDP__nnUNetPlansv2.1"
# --------------------------------------------------------------------------

# Step 1 — Convert input MHA files to nnUNet raw-data format
python nnunet/dataset_conversion/Task500_Ischemic_Stroke_Test.py

# Step 2 — Rewrite DDP trainer class in checkpoint pkl files so that
#           single-GPU nnUNet_predict can load them.
#           Comment out this step when using a single-GPU-trained model.
nnUNet_change_trainer_class \
    -i $MODEL_DIR \
    -tr nnUNetTrainerV2

# Step 3 — Run inference for each of the 5 cross-validation folds
CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold0 \
               -t $TASK_ID \
               -tr $TRAINER \
               -m 3d_fullres \
               -f 0 \
               -z \
               --disable_postprocessing

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold1 \
               -t $TASK_ID \
               -tr $TRAINER \
               -m 3d_fullres \
               -f 1 \
               -z \
               --disable_postprocessing

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold2 \
               -t $TASK_ID \
               -tr $TRAINER \
               -m 3d_fullres \
               -f 2 \
               -z \
               --disable_postprocessing

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold3 \
               -t $TASK_ID \
               -tr $TRAINER \
               -m 3d_fullres \
               -f 3 \
               -z \
               --disable_postprocessing

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold4 \
               -t $TASK_ID \
               -tr $TRAINER \
               -m 3d_fullres \
               -f 4 \
               -z \
               --disable_postprocessing

# Step 4 — Recover softmax predictions to the original image space
python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold0 \
                        -m preliminary_phase \
                        -f fold0

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold1 \
                        -m preliminary_phase \
                        -f fold1

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold2 \
                        -m preliminary_phase \
                        -f fold2

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold3 \
                        -m preliminary_phase \
                        -f fold3

python recover_softmax.py \
                        -i test_result \
                        -o test_result_recover/preliminary_phase/fold4 \
                        -m preliminary_phase \
                        -f fold4

# Step 5 — Ensemble the 5 folds by averaging softmax probabilities
model_0=test_result_recover/preliminary_phase/fold0
model_1=test_result_recover/preliminary_phase/fold1
model_2=test_result_recover/preliminary_phase/fold2
model_3=test_result_recover/preliminary_phase/fold3
model_4=test_result_recover/preliminary_phase/fold4

python -m ensemble_predictions \
           -f $model_0 \
              $model_1 \
              $model_2 \
              $model_3 \
              $model_4 \
           -o test_ensemble/ \
           --npz

# Step 6 — Apply threshold and write final segmentation to output directory
python threshold_redirect.py \
                            -i test_ensemble/ \
                            -o output/images/stroke-lesion-segmentation/
