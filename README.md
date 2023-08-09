# Biendata-URISC

This is a course project which also aims at solving Ultra-high Resolution EM Images Segmentation Challenge in the Biendata competition
detailed information and training datasets can be found in [urisc-Biendata](https://www.biendata.net/competition/urisc/)

This project is mainly designed for the complex track, but can also be used in the simple track. It is mainly composed of three parts

## Part 1 data preprocessing

The raw dataset in the complex track only contains 20 10000 $\times$ 10000 images. So we use random crop and sliding window algorithms to split the raw images into small parts
Then we further apply random erasing, $\gamma$ transform, and other data augmentation technics to enlarge the dataset

## Part 2 model training

model           LinkNet
Loss function   Focal loss + Dice loss

## Part 3 postprocessing

we split the large test images piece by piece, and then concatenate partial results to get the final inference
we use TTA to make more confidential predictions. We predict the raw image, and its transposition and flip together. Then we integrate the prediction result

## methods that seem reasonable but don't work for us

two-stage training

Morphological opening operations






