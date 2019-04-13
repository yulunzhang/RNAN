# EDSR baseline model (x2)
#python main.py --model EDSR --scale 2 --save EDSR_baseline_x2 --reset

# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --save EDSR_baseline_x3 --reset --pre_train ../experiment/model/EDSR_baseline_x2.pt

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save EDSR_baseline_x4 --reset --pre_train ../experiment/model/EDSR_baseline_x2.pt

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save EDSR_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save EDSR_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train ../experiment/EDSR_x2/model/model_best.pt

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save EDSR_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train ../experiment/EDSR_x2/model/model_best.pt

# MDSR baseline model
CUDA_VISIBLE_DEVICES=1 python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_results

# MDSR in the paper
CUDA_VISIBLE_DEVICES=2 python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_results
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble
#python main.py --data_test Set14 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble
#python main.py --data_test B100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble
#python main.py --data_test Urban100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble
#python main.py --data_test DIV2K --ext img --n_val 100 --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --self_ensemble

#python main.py --data_test Set5 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test Set14 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test B100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test Urban100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble
#python main.py --data_test DIV2K --ext img --n_val 100 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/model/EDSR_x4.pt --test_only --self_ensemble

# Test your own images
python main.py --data_test Demo --scale 4 --pre_train ../experiment/model/EDSR_baseline_x4.pt --test_only --save_results

# Advanced - Test with JPEG images 
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train ../experiment/model/MDSR_baseline_jpeg.pt --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save EDSR_GAN --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train ../experiment/model/EDSR_baseline_x4.pt

# For ECCV-2018 RCAN rebuttal
LOG=./../experiment/MRCAN_G10R20P48B16BIX2DIV2K-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=1 python main.py --model MRCAN --scale 2+3+4 --save MRCAN_G10R20P48B16BIX2DIV2K --reset --save_results --chop --print_model --n_resgroups 10 --n_resblocks 20 --n_feats 64 --patch_size 12  2>&1 | tee $LOG

# ablation study
# RCAN_G10R20
LOG=./../experiment/RCAN_G10R20P48B16BIX2DIV2K-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=1 python main.py --model RIRSESR --scale 2 --save RCAN_G10R20P48B16BIX2DIV2K --reset --save_results --chop --print_model --n_resgroups 10 --n_resblocks 20 --n_feats 64 --patch_size 96  2>&1 | tee $LOG
# RCAN_G6R20
LOG=./../experiment/RCAN_G6R20P48B16BIX2DIV2K-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=2 python main.py --model RIRSESR --scale 2 --save RCAN_G6R20P48B16BIX2DIV2K --reset --save_results --chop --print_model --n_resgroups 6 --n_resblocks 20 --n_feats 64 --patch_size 96  2>&1 | tee $LOG
# RCAN_G4R20
LOG=./../experiment/RCAN_G4R20P48B16BIX2DIV2K-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=2 python main.py --model RIRSESR --scale 2 --save RCAN_G4R20P48B16BIX2DIV2K --reset --save_results --chop --print_model --n_resgroups 4 --n_resblocks 20 --n_feats 64 --patch_size 96  2>&1 | tee $LOG
# RCAN_G10R10
LOG=./../experiment/RCAN_G10R10P48B16BIX2DIV2K-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=1 python main.py --model RIRSESR --scale 2 --save RCAN_G10R10P48B16BIX2DIV2K --reset --save_results --chop --print_model --n_resgroups 10 --n_resblocks 10 --n_feats 64 --patch_size 96  2>&1 | tee $LOG
# RCAN_G10R15
LOG=./../experiment/RCAN_G10R15P48B16BIX2DIV2K-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=2 python main.py --model RIRSESR --scale 2 --save RCAN_G10R15P48B16BIX2DIV2K --reset --save_results --chop --print_model --n_resgroups 10 --n_resblocks 15 --n_feats 64 --patch_size 96  2>&1 | tee $LOG


# train on SR+Deblur data
# RCAN_SR_Deblur_G10R20P48B16X2DIV2K
LOG=./../experiment/RCAN_SR_Deblur_G10R20P48B16X2DIV2K-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=3 python main.py --model RIRSESR --scale 2 --save RCAN_SR_Deblur_G10R20P48B16X2DIV2K --reset --save_results --chop --print_model --n_resgroups 10 --n_resblocks 20 --n_feats 64 --patch_size 96  --dir_data /home/yulun/data/SR_Deblur 2>&1 | tee $LOG

# EDSR_SR_Deblur_R80F64P48X2DIV2K
LOG=./../experiment/EDSR_SR_Deblur_R80F64P48X2DIV2K-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model EDSR --scale 2 --save EDSR_SR_Deblur_R80F64P48X2DIV2K --reset --save_results --chop --print_model --n_resblocks 80 --n_feats 64 --patch_size 96  --dir_data /home/yulun/data/SR_Deblur 2>&1 | tee $LOG

# investigate reduction in CA module
LOG=./../experiment/RCAN_G10R10P48B16BIX2DIV2K_reduc4-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=1 python main.py --model RIRSESR --scale 2 --save RCAN_G10R10P48B16BIX2DIV2K_reduc4 --reset --save_results --chop --print_model --n_resgroups 10 --n_resblocks 10 --n_feats 64 --patch_size 96 --reduction 4 2>&1 | tee $LOG

LOG=./../experiment/RCAN_G10R10P48B16BIX2DIV2K_NODU-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model RIRSESRNODU --scale 2 --save RCAN_G10R10P48B16BIX2DIV2K_NODU --reset --save_results --chop --print_model --n_resgroups 10 --n_resblocks 10 --n_feats 64 --patch_size 96 2>&1 | tee $LOG

LOG=./../experiment/RCAN_G10R10P48B16BIX2DIV2K_NORELU-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=2 python main.py --model RIRSESRNORELU --scale 2 --save RCAN_G10R10P48B16BIX2DIV2K_NORELU --reset --save_results --chop --print_model --n_resgroups 10 --n_resblocks 10 --n_feats 64 --patch_size 96 2>&1 | tee $LOG


















































