# VoteHMR
the official implementation of ACM MM 2021 paper <VoteHMR:Occlusion-Aware Voting Network for Robust 3D Human Mesh Recovery from Partial Point Clouds>
# Introduction
This is a release of our paper <VoteHMR:Occlusion-Aware Voting Network for Robust 3D Human Mesh Recovery from Partial Point Clouds>  
Authors: Guanze Liu, Yu Rong, Lu Sheng*  
[[arxiv]](https://arxiv.org/abs/2110.08729)
# Citation
> @inproceedings{liu2021votehmr,  
  title={VoteHMR: Occlusion-Aware Voting Network for Robust 3D Human Mesh Recovery from Partial Point Clouds},  
  author={Liu, Guanze and Rong, Yu and Sheng, Lu},  
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},  
  pages={955--964},  
  year={2021}  
}
# Installation
The code is tested under the following environment
* pytorch 1.1.0
* python 3.7.10
* cuda 10.1
* pointnet2 (https://github.com/facebookresearch/votenet/tree/main/pointnet2)
# Data Preparation
## Download Human Models
the required SMPL models are uploaded to [BaiduYUN](https://pan.baidu.com/s/19D4WGM1-bhRR-06iAO5l_A) [extract code: ame2] for download  
## Set up Blender

You need to download Blender and install scipy package to run the first part of the code. The provided code was tested with [Blender2.78](http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2), which is shipped with its own python executable as well as distutils package. Therefore, it is sufficient to do the following:
* Install pip
> /blenderpath/2.78/python/bin/python3.5m get-pip.py
* Install scipy
> /blenderpath/2.78/python/bin/python3.5m pip install scipy
* Install numpy
> /blenderpath/2.78/python/bin/python3.5m pip install numpy  

Notice that `get-pip.py` is downloaded from [pip](https://pip.pypa.io/en/stable/installing/). Replace the `blenderpath` with your own and set `BLENDER_PATH`.

## Set up OpenEXR

In order to read rendered depth images and segm images, the [OpenEXR bindings for PYTHON](http://www.excamera.com/sphinx/articles-openexr.html) is required.  
Set the `openexr_py2_path` in `src/datasets/config.copy` as your OpenEXR path.

## Prepare Training Data
The Synthetic Dataset [SURREAL](https://github.com/gulvarol/surreal) and [DFAUST](https://dfaust.is.tue.mpg.de/) can be downloaded from existing repositary.  
We provide the shell script to generate partial point cloud from provided synthetic data.  
You should set the corresponding data_path in the following shell scripts, including `surreal_data_path`, `surreal_save_path`, `dfaust_save_path`, as well as the `tmp_path`, `output_path` in `src/datasets/config.copy`
> sh scripts/generate_training_data.sh
## Prepare Testing Data
We also provide the shell script to generate testing data
> sh scripts/generate_testing_data.sh
# Training
We provide the shell scripts to train with VoteHMR  
* If you wish to train on single gpu, run the following shell script 
> sh scripts/train.sh
* If you wish to train on multiple gpu, run the following shell script
> sh scripts/train_dist.sh
# Evaluation
We provide the shell scripts for evaluation, set chechpoints as your model path dir.
> sh scripts/test.sh
# Fine Tuning on Real Data
If you wish to fine tune on real data, run  
> sh scripts/weakly_supervised.sh

# For Online Visdom Support
Before training starts, to visualize the training results and loss curve in real-time, please run `python -m visdom.server 8098` and click the URL `https//localhost:8098`