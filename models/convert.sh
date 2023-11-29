#!/bin/bash
  
dataset="../dataset/test/"
mlir_model="sr_model_batch_400_tile_10_10.mlir"
calibration_table="sr_model_batch_400_tile_10_10_table"
out_bmodel="out.bmodel"

# model transform
model_transform.py --model_name sr-model --input_shape [[400,3,10,10]] --model_def sr_model.pt --mean 0.0,0.0,0.0 --scale 0.0039216,0.0039216,0.0039216 --mlir $mlir_model

# run calibration
run_calibration.py $mlir_model --dataset $dataset --input_num 600 -o $calibration_table --tune_num 5

# model deploy
model_deploy.py --mlir $mlir_model --quantize INT8 --chip bm1684x --model $out_bmodel --calibration_table $calibration_table

#  --fuse_preprocess