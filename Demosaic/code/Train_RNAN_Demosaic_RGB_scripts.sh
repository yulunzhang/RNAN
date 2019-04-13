#!/bin/bash/
# pytorch 0.3.1
# train scripts
# RNAN_F64G10P48L2N1: N1 means case 1
python main.py --model RNAN --noise_level 1 --save RNAN_Demosaic_RGB_F64G10P48L2N1 --patch_size 48 --save_results --chop --loss 1*MSE  
