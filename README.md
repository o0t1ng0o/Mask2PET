# Mask2PET
This repository is a PyTorch implementation for PET tumor generation from benign PET and tumor mask. We build this code upon [medfusion](https://github.com/mueller-franzes/medfusion). You can completely follow the instruction in medfusion.

## Dataset
We use the HECKTOR 2021 dataset, please download from this link (https://www.aicrowd.com/challenges/miccai-2021-hecktor). Please check the path of data in ./launch/train.sh and ./launch/test.sh

## Data preprocessing
In this work, we use the VQ-GAN as encoder to encode the PET. If you want to custom this model on your dataset, please train VQ-GAN on your dataset first. https://github.com/CompVis/taming-transformers.

We provide the pre-trained VQ-GAN model in ./pretrained_models

## Training 
```
sh ./launch/train.sh
```
## inference
```
sh ./launch/test.sh
```

