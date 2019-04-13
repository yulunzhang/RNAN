#!/bin/bash/
# pytorch 0.3.1
# test scripts
# No self-ensemble, use different testsets (Classic5, LIVE1) to reproduce the results in the paper.
# Q = 10
python main.py --model RNAN --data_test Demo --noise_level 10 --save Test_RNAN --save_results --chop --test_only  --pre_train ../experiment/model/RNAN_CAR_Y_F64G10P48L2N10.pt --testpath ../experiment/LQ --testset Classic5
# Q = 20
python main.py --model RNAN --data_test Demo --noise_level 20 --save Test_RNAN --save_results --chop --test_only  --pre_train ../experiment/model/RNAN_CAR_Y_F64G10P48L2N20.pt --testpath ../experiment/LQ --testset Classic5
# Q = 30
python main.py --model RNAN --data_test Demo --noise_level 30 --save Test_RNAN --save_results --chop --test_only  --pre_train ../experiment/model/RNAN_CAR_Y_F64G10P48L2N30.pt --testpath ../experiment/LQ --testset Classic5
# Q = 40
python main.py --model RNAN --data_test Demo --noise_level 40 --save Test_RNAN --save_results --chop --test_only  --pre_train ../experiment/model/RNAN_CAR_Y_F64G10P48L2N40.pt --testpath ../experiment/LQ --testset Classic5

