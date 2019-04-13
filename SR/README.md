# Residual Non-local Attention Networks for Image Restoration
This repository is for RNAN introduced in the following paper

[Yulun Zhang](http://yulunzhang.com/), [Kunpeng Li](https://kunpengli1994.github.io/), [Kai Li](http://kailigo.github.io/), [Lichen Wang](https://sites.google.com/site/lichenwang123/), [Bineng Zhong](https://scholar.google.de/citations?user=hvRBydsAAAAJ&hl=en), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/), "Residual Non-local Attention Networks for Image Restoration", ICLR 2019, [[OpenReview]](https://openreview.net/pdf?id=HkeGhoA5FX) 

[SR] The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 14.04/16.04 environment (Python3.6, PyTorch_0.4.0, CUDA8.0, cuDNN5.1) with Titan X/1080Ti/Xp GPUs.

## Contents
1. [Train](#train)
2. [Test](#test)
3. [Results](#results)
4. [Citation](#citation)
5. [Acknowledgements](#acknowledgements)

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

1. (optional) Download models for our paper and place them in '/RNAN/[Task]/experiment/model'. [Task] means CAR, DN_RGB, DN_Gray, Demosaic, or SR.

    All the models can be downloaded from [Github](https://github.com/yulunzhang/modelzoo/tree/master/RNAN).

2. Cd to 'RNAN/[Task]/code', run the following scripts to train models.

    **You can use scripts in file 'Train_RNAN_scripts' to train models for our paper.**

    ```bash
    # pytorch0.4.0, cuda8.0,
    # source activate pytorch040
    # RNAN_SR_F64G10P48BIX2
    CUDA_VISIBLE_DEVICES=1 python main.py --model RNAN --scale 2 --save RNAN_SR_F64G10P48BIX2 --save_results --chop --patch_size 96
    # RNAN_SR_F64G10P48BIX3
    CUDA_VISIBLE_DEVICES=1 python main.py --model RNAN --scale 3 --save RNAN_SR_F64G10P48BIX3 --save_results --chop --patch_size 144
    # RNAN_SR_F64G10P48BIX4
    CUDA_VISIBLE_DEVICES=1 python main.py --model RNAN --scale 4 --save RNAN_SR_F64G10P48BIX4 --save_results --chop --patch_size 192 

    ```
## Test
### Quick start
1. Download models for our paper and place them in '/RNAN/[Task]/experiment/model'. [Task] means CAR, DN_RGB, DN_Gray, Demosaic, or SR.

    All the models can be downloaded from [Github](https://github.com/yulunzhang/modelzoo/tree/master/RNAN).

2. Cd to 'RNAN/[Task]/code', run the following scripts.

    **You can use scripts in file 'Test_RNAN_scripts' to produce results for our paper.**

    ```bash
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
    ```

### The whole test pipeline
1. Prepare test data.

    Place the original test sets (e.g., Set5, other test sets are available from [GoogleDrive](https://drive.google.com/drive/folders/1xyiuTr6ga6ni-yfTP7kyPHRmfBakWovo?usp=sharing) or [Baidu](https://pan.baidu.com/s/1yBI_-rknXT2lm1UAAB_bag)) in 'OriginalTestData'.

    Run 'Prepare_TestData_HR_LR.m' in Matlab to generate HR/LR images with different degradation models.
2. Conduct image SR. 

    See **Quick start**
3. Evaluate the results.

    Run 'Evaluate_PSNR_SSIM.m' to obtain PSNR/SSIM values for paper.

## Results
### Quantitative Results
### Visual Results

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}

@inproceedings{zhang2019rnan,
    title={Residual Non-local Attention Networks for Image Restoration},
    author={Zhang, Yulun and Li, Kunpeng and Li, Kai and Zhong, Bineng and Fu, Yun},
    booktitle={ICLR},
    year={2019}
}
```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes of EDSR [Torch version](https://github.com/LimBee/NTIRE2017) and [PyTorch version](https://github.com/thstkdgus35/EDSR-PyTorch).
