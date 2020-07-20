# FaceSwapping

## Overview

This page contains code related to the following two papers:

- Bailer, Werner. "Face swapping for solving collateral privacy issues in multimedia analytics." International Conference on Multimedia Modeling. Springer, 2019. [Link](https://link.springer.com/chapter/10.1007/978-3-030-05710-7_14)

- Bailer, Werner, and Martin Winter. "On Improving Face Generation for Privacy Preservation." 2019 International Conference on Content-Based Multimedia Indexing (CBMI). IEEE, 2019. [PDF](https://www.projectmarconi.eu/s/facegen.pdf)

The pipeline contains of the follwing steps:

- (optional) segmentation of training data of the GAN
- GAN for generating a set of faces to be used for replacement
- swapping: detect faces in the provided image and one of the generated face images, align them and replace the faces in the provided image (this step does not need training)

## Segmentation (optional)

In order to avoid issues with the diverse backgrounds, images are first segmented before using them to train the GAN for face generation.

The segmenter code in automatic-portraint-tf is forked from [here](https://github.com/Corea/automatic-portrait-tf)

The pretrained `FCN8s model` is used as a starting point. A trained model can be downloaded from [here](https://faceswapping.s3-eu-west-1.amazonaws.com/automatic-portrait-tf-master/model/model.zip)

## Face generation

The GAN code in `DCGAN-tensorflow` is forked from [here](https://github.com/carpedm20/DCGAN-tensorflow)

Models:
- [model trained on CelebA (without segmentation, image size 108)] (https://github.com/bamos/dcgan-completion.tensorflow/tree/master/checkpoint)
- [model trained on CelebA (with segmentation, image size 216)](https://faceswapping.s3-eu-west-1.amazonaws.com/DCGAN-tensorflow/checkpoint/celebAseg_64_216_216.zip)

## Face swapping

The code in `tbd` is a standalone C++ application (project for MSVC 2017 is provided). 
