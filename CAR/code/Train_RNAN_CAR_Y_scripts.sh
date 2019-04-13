#!/bin/bash/
# pytorch 0.3.1
# train scripts
# RNAN_CAR_Y_F64G10P48L2N10
python main.py --model RNAN --noise_level 10 --save RNAN_CAR_Y_F64G10P48L2N10 --patch_size 48 --save_results --chop --loss 1*MSE
# RNAN_CAR_Y_F64G10P48L2N20
python main.py --model RNAN --noise_level 20 --save RNAN_CAR_Y_F64G10P48L2N20 --patch_size 48 --save_results --chop --loss 1*MSE
# RNAN_CAR_Y_F64G10P48L2N30
python main.py --model RNAN --noise_level 30 --save RNAN_CAR_Y_F64G10P48L2N30 --patch_size 48 --save_results --chop --loss 1*MSE
# RNAN_CAR_Y_F64G10P48L2N40
python main.py --model RNAN --noise_level 40 --save RNAN_CAR_Y_F64G10P48L2N40 --patch_size 48 --save_results --chop --loss 1*MSE


