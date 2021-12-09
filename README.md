# Finetuning and Evaluating Vision Transformers and ResNets for Image Classification

### Validating image classification benchmark results on ViTs, DeiT and ResNets BiT models
---


Hello everyone!
This repository contains the code implementation for our ML701 project "Transformers Transforming Vision", in which we have revisted the baseline vision transformers such as ViTs, DeiT and compared them with the CNN models like ResNet BiT.

We have tried to validate the official published results of ViTs, DeiTs and ResNets on CIFAR-10 and CIFAR-100, when they are pretrained on different datasets like ImageNet and ImageNet-21k. Additionally, we have used CUB200 dataset and exploited the importance of having high resolution images specifically for vision transformers. Our approch is summarized below:
<p align="center">
  <img src="extras/approach.png" width="700" height="400">
</p>

Specifically, we provide finetuning and evalutation scripts for all ResNet-BiT, ViT and DeiT models which are supported by PyTorch library.



-----------

Requirements
---
To run the scripts, following packages needs to be installed (preferably on Ubuntu 18.04 LTS / 20.04 LTS):
<ul>
  <li>
    Python (version 3.6 or greater)
  </li>
    <li>
    Pytorch (version 1.10) and Torchvision (version 0.3.0)
  </li>
    <li>
    Pytorch timm library (preferably version 0.4.12)
    </li>
  </ul>
  
  
 To install these, use pip package installer and execute the following commands:

  For CUDA-10.2, install pytorch and torchvision as follows:
  ```bash
 $ pip3 install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```
  ```bash
$ pip3 install pip install timm
```
------------

Finetuning models
---


------------



Evaluating model
---



------------
