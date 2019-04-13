#!/bin/bash
# pytorch 0.3.1
# test scripts
# No self-ensemble, use different testsets (Kodak24, CBSD68, McMaster18, Urban100) to reproduce the results in the paper.
# case 1
python main.py --model RNAN --data_test Demo --noise_level 1 --save Test_RNAN --n_cab_1 20 --save_results --test_only --chop --pre_train ../experiment/model/RNAN_Demosaic_RGB_F64G10P48L2N1.pt --testpath ../experiment/LQ --testset Kodak24


