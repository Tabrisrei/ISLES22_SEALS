export nnUNet_raw_data_base="data/nnUNet_raw_data_base"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export RESULTS_FOLDER="data/nnUNet_trained_models"
export nnUNet_n_proc_DA=24

python nnunet/dataset_conversion/Task500_Ischemic_Stroke_Test.py

nnUNet_change_trainer_class -i data/nnUNet_trained_models/nnUNet/3d_fullres/Task012_Ischemic_Stroke_TM_Fullset/nnUNetTrainerV2_DDP__nnUNetPlansv2.1 \
                            -tr nnUNetTrainerV2 \

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold0 \
               -t 12 \
               -tr nnUNetTrainerV2_DDP \
               -m 3d_fullres \
               -f 0 \
               -z \
               --disable_postprocessing  

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold1 \
               -t 12 \
               -tr nnUNetTrainerV2_DDP \
               -m 3d_fullres \
               -f 1 \
               -z \
               --disable_postprocessing

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold2 \
               -t 12 \
               -tr nnUNetTrainerV2_DDP \
               -m 3d_fullres \
               -f 2 \
               -z \
               --disable_postprocessing
               
CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold3 \
               -t 12 \
               -tr nnUNetTrainerV2_DDP \
               -m 3d_fullres \
               -f 3 \
               -z \
               --disable_postprocessing

CUDA_VISIBLE_DEVICES=0 \
nnUNet_predict \
               -i $nnUNet_raw_data_base/nnUNet_raw_data/Task500_Ischemic_Stroke_Test/imagesTs/ \
               -o test_result/preliminary_phase/fold4 \
               -t 12 \
               -tr nnUNetTrainerV2_DDP \
               -m 3d_fullres \
               -f 4 \
               -z \
               --disable_postprocessing

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

model_0=test_result_recover/preliminary_phase/fold0
model_1=test_result_recover/preliminary_phase/fold1
model_2=test_result_recover/preliminary_phase/fold2
model_3=test_result_recover/preliminary_phase/fold3
model_4=test_result_recover/preliminary_phase/fold4
python -m ensemble_predictions -f $model_0 \
                                  $model_1 \
                                  $model_2 \
                                  $model_3 \
                                  $model_4 \
                                -o test_ensemble/ \
                                --npz

python threshold_redirect.py \
                            -i test_ensemble/ \
                            -o output/images/stroke-lesion-segmentation/

