#!/bin/bash/
# pytorch 0.3.1
# train scripts
# RNAN_F64G10P48L2N10
python main.py --model RNAN --noise_level 10 --save RNAN_Gray_F64G10P48L2N10 --patch_size 48  --save_results --chop --loss 1*MSE
# RNAN_F64G10P48L2N30
python main.py --model RNAN --noise_level 30 --save RNAN_Gray_F64G10P48L2N30 --patch_size 48  --save_results --chop --loss 1*MSE
# RNAN_F64G10P48L2N50
python main.py --model RNAN --noise_level 50 --save RNAN_Gray_F64G10P48L2N50 --patch_size 48  --save_results --chop --loss 1*MSE
# RNAN_F64G10P48L2N70
python main.py --model RNAN --noise_level 70 --save RNAN_Gray_F64G10P48L2N70 --patch_size 48  --save_results --chop --loss 1*MSE



















