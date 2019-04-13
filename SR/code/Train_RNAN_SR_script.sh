#!/bin/bash/
# pytorch0.4.0, cuda8.0,
# source activate pytorch040
# RNAN_SR_F64G10P48BIX2
CUDA_VISIBLE_DEVICES=1 python main.py --model RNAN --scale 2 --save RNAN_SR_F64G10P48BIX2 --save_results --chop --patch_size 96
# RNAN_SR_F64G10P48BIX3
CUDA_VISIBLE_DEVICES=1 python main.py --model RNAN --scale 3 --save RNAN_SR_F64G10P48BIX3 --save_results --chop --patch_size 144
# RNAN_SR_F64G10P48BIX4
CUDA_VISIBLE_DEVICES=1 python main.py --model RNAN --scale 4 --save RNAN_SR_F64G10P48BIX4 --save_results --chop --patch_size 192 



