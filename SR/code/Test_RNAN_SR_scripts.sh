#!/bin/bash/
# pytorch0.4.0, cuda8.0,
# test scripts
# No self-ensemble, use different testsets (Set5, Set14, B100, Urban100, Manga109) to reproduce the results in the paper.
# X2
CUDA_VISIBLE_DEVICES=3 python main.py --model RNAN --data_test Demo --scale 2 --save Test_RNAN --save_results --test_only --chop --pre_train ../experiment/model/RNAN_SR_F64G10P48BIX2.pt --testpath ../experiment/LR --testset Set5
# X3
CUDA_VISIBLE_DEVICES=2 python main.py --model RNAN --data_test Demo --scale 3 --save Test_RNAN --save_results --test_only --chop --pre_train ../experiment/model/RNAN_SR_F64G10P48BIX3.pt --testpath ../experiment/LR --testset Set5
# X4
CUDA_VISIBLE_DEVICES=1 python main.py --model RNAN --data_test Demo --scale 4 --save Test_RNAN --save_results --test_only --chop --pre_train ../experiment/model/RNAN_SR_F64G10P48BIX4.pt --testpath ../experiment/LR --testset Set5

# use self-ensemble
# X2
CUDA_VISIBLE_DEVICES=3 python main.py --model RNAN --data_test Demo --scale 2 --save Test_RNANplus --save_results --test_only --chop --pre_train ../experiment/model/RNAN_SR_F64G10P48BIX2.pt --self_ensemble --testpath ../experiment/LR --testset Set5
# X3
CUDA_VISIBLE_DEVICES=2 python main.py --model RNAN --data_test Demo --scale 3 --save Test_RNANplus --save_results --test_only --chop --pre_train ../experiment/model/RNAN_SR_F64G10P48BIX3.pt --self_ensemble --testpath ../experiment/LR --testset Set5
# X4
CUDA_VISIBLE_DEVICES=1 python main.py --model RNAN --data_test Demo --scale 4 --save Test_RNANplus --save_results --test_only --chop --pre_train ../experiment/model/RNAN_SR_F64G10P48BIX4.pt --self_ensemble --testpath ../experiment/LR --testset Set5




