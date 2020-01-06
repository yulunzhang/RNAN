# Residual Non-local Attention Networks for Image Restoration
This repository is for RNAN introduced in the following paper

[Yulun Zhang](http://yulunzhang.com/), [Kunpeng Li](https://kunpengli1994.github.io/), [Kai Li](http://kailigo.github.io/), [Bineng Zhong](https://scholar.google.de/citations?user=hvRBydsAAAAJ&hl=en), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/), "Residual Non-local Attention Networks for Image Restoration", ICLR 2019, [[OpenReview]](https://openreview.net/pdf?id=HkeGhoA5FX) 

The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 14.04/16.04 environment (Python3.6, PyTorch_0.3.1 for DN/CAR/Demosaic, PyTorch_0.4.0 for SR, CUDA8.0, cuDNN5.1) with Titan X/1080Ti/Xp GPUs.

## Contents
1. [Introduction](#Introduction)
2. [Tasks](#Tasks)
3. [Citation](#citation)
4. [Acknowledgements](#acknowledgements)

## Introduction
In this paper, we propose a residual non-local attention network for high-quality image restoration. Without considering the uneven distribution of information in the corrupted images, previous methods are restricted by local convolutional operation and equal treatment of spatial- and channel-wise features. To address this issue, we design local and non-local attention blocks to extract features that capture the long-range dependencies between pixels and pay more attention to the challenging parts. Specifically, we design trunk branch and (non-)local mask branch in each (non-)local attention block. The trunk branch is used to extract hierarchical features. Local and non-local mask branches aim to adaptively rescale these hierarchical features with mixed attentions. The local mask branch concentrates on more local structures with convolutional operations, while non-local attention considers more about long-range dependencies in the whole feature map. Furthermore, we propose residual local and non-local attention learning to train the very deep network, which further enhance the representation ability of the network. Our proposed method can be generalized for various image restoration applications, such as image denoising, demosaicing, compression artifacts reduction, and super-resolution. Experiments demonstrate that our method obtains comparable or better results compared with recently leading methods quantitatively and visually. 
![block](/Figs/block.PNG)

## Tasks
### Gray-scale Image Denoising 
![PSNR_DN_Gray](/Figs/PSNR_DN_Gray.PNG)
![Visual_DN_Gray](/Figs/Visual_DN_Gray.PNG)
More details at [DN_Gray](https://github.com/yulunzhang/RNAN/tree/master/DN_Gray).
### Color Image Denoising 
![PSNR_DN_RGB](/Figs/PSNR_DN_RGB.PNG)
![Visual_DN_RGB](/Figs/Visual_DN_RGB.PNG)
More details at [DN_RGB](https://github.com/yulunzhang/RNAN/tree/master/DN_RGB).
### Image Demosaicing 
![PSNR_Demosaic](/Figs/PSNR_Demosaic.PNG)
![Visual_Demosaic](/Figs/Visual_Demosaic.PNG)
More details at [Demosaic](https://github.com/yulunzhang/RNAN/tree/master/Demosaic).
### Image Compression Artifact Reduction 
![PSNR_CAR](/Figs/PSNR_CAR.PNG)
![Visual_CAR](/Figs/Visual_CAR.PNG)
More details at [CAR](https://github.com/yulunzhang/RNAN/tree/master/CAR).
### Image Super-resolution 
![PSNR_SR](/Figs/PSNR_SR.PNG)
![Visual_SR](/Figs/Visual_SR.PNG)
More details at [SR](https://github.com/yulunzhang/RNAN/tree/master/SR).

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
