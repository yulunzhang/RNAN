#!/bin/bash
# pytorch 0.3.1
# test scripts
# No self-ensemble, use different testsets (Kodak24, CBSD68, Urban100) to reproduce the results in the paper.
# N=10
python main.py --model RNAN --data_test Demo --noise_level 10 --save Test_RNAN --save_results --test_only --chop --pre_train ../experiment/model/RNAN_F64G10P48L2N10.pt --testpath ../experiment/LQ --testset Kodak24
# N=30
python main.py --model RNAN --data_test Demo --noise_level 30 --save Test_RNAN --save_results --test_only --chop --pre_train ../experiment/model/RNAN_F64G10P48L2N30.pt --testpath ../experiment/LQ --testset Kodak24
# N=50
python main.py --model RNAN --data_test Demo --noise_level 50 --save Test_RNAN --save_results --test_only --chop --pre_train ../experiment/model/RNAN_F64G10P48L2N50.pt --testpath ../experiment/LQ --testset Kodak24
# N=70
python main.py --model RNAN --data_test Demo --noise_level 70 --save Test_RNAN --save_results --test_only --chop --pre_train ../experiment/model/RNAN_F64G10P48L2N70.pt --testpath ../experiment/LQ --testset Kodak24



