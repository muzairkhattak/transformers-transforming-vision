#importing basic libraries
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import ImageFile
# from efficientnet_pytorch import EfficientNet
ImageFile.LOAD_TRUNCATED_IMAGES = True

#custom libraries
from models import Net,VisionTransformers,ResNet152,BiT
from train import train_model
import my_transforms
from my_datasets import CIFAR10, CIFAR100
import utils
import test
from config import args

#experiment (1): use ViT imagenet1k on both CIFAR10 and CIFAR100 (fine tuning)
# model = VisionTransformers(10,'vit_base_patch16_224').return_model()
#experiment (2): use ViT imagenet-21k on both CIFAR10 and CIFAR100 (fine tuning)
# model = VisionTransformers(10,'vit_base_patch16_224_in21k').return_model()
#experiment (3): use ResNet101 BiT imagenet1k on both CIFAR10 and CIFAR100 (fine tuning)
# model=BiT(10,'resnetv2_101x3_bitm').return_model()
#experiment (4): use ResNet101 BiT imagenet-21k on both CIFAR10 and CIFAR100 (fine tuning)
# model=BiT(10,'resnetv2_101x3_bitm_in21k').return_model()
#experiment (5): use DeiT Base  imagenet1k on both CIFAR10 and CIFAR100 (fine tuning)
# model=VisionTransformers(10,'resnetv2_101x3_bitm_in21k').return_model()
#experiment (6): use DeiT Distilled Base imagenet1k on both CIFAR10 and CIFAR100 (fine tuning)
# model=VisionTransformers(10,'resnetv2_101x3_bitm_in21k').return_model()


models_list=['resnetv2_101x3_bitm','resnetv2_101x3_bitm_in21k']
Args=args()
if Args.training==True:

    for model_name in Args.model_list:
        model=None
        if model_name in models_list:
            model=BiT(Args.classes,model_name).return_model()
        else:
            model=VisionTransformers(Args.classes,model_name).return_model()

        # define the loss function
        optimizer = optim.Adam(model.parameters(), lr = Args.lr)
        num_epochs=50
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.05)
        print("Training model : ", model_name)
        name=model_name+ "_" + str(num_epochs)+ "_" + str(Args.batch_size)+ "CIFAR100"
        #create a dataloader and choose a dataset
        trainloader,testloader,dataset_sizes=CIFAR100(my_transforms.CIFAR100_transform(),my_transforms.CIFAR100_transform('val'),batch_size=Args.batch_size,val_pct=Args.val_pct).return_dataloader()
        dataloader = {'train': trainloader, 'val': testloader}

        model, info = train_model(model,model_name, dataloader,dataset_sizes, Args.criterion, optimizer,scheduler,num_epochs=num_epochs)

        #save the model onto disk
        torch.save(model.state_dict(), Args.base_PATH+ name)
        #visualize the training and validation results
        utils.show_info(info, model_name,Args.figure_path+name)
        utils.show_info_loss(info, model_name,Args.figure_path+name)
        testloader,dataset_sizes=CIFAR100(my_transforms.CIFAR100_transform(),my_transforms.CIFAR100_transform('val'),batch_size=Args.batch_size,return_test_dataloader=True).return_dataloader()
        test.test_model(model,testloader,dataset_sizes)


else:  #just load the trained weights and evaluate on the test dataset

        for model_name in Args.model_list:
            model=None
            if model_name in models_list:
                model=BiT(Args.classes,model_name).return_model()
            else:
                model=VisionTransformers(Args.classes,model_name).return_model()

            model.load_state_dict(torch.load(Args.model_PATH))
            testloader,dataset_sizes=CIFAR10(my_transforms.CIFAR10_transform(),my_transforms.CIFAR10_transform('val'),batch_size=Args.batch_size,return_test_dataloader=True).return_dataloader()
            test.test_model(model,testloader,dataset_sizes)
    # if training==True:`

#     #lets make a loop 
#     batch_size_list=[1,2,4,16,32,64]
#         # define the optimizer
#     criterion = nn.CrossEntropyLoss()
#         # define the loss function
#     optimizer = optim.Adam(model.parameters(), lr = lr)
#     num_epochs=30
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.05)
#     for batch_size in batch_size_list:
#         print("Training for batch size ", batch_size)
#         name="ResNet152_"+str(num_epochs)+ "_" + str(batch_size)
#         #create a dataloader and choose a dataset
#         trainloader,testloader,dataset_sizes=CIFAR10(my_transforms.CIFAR10_transform(),my_transforms.CIFAR10_transform('val'),batch_size=batch_size).return_dataloader()
#         dataloader = {'train': trainloader, 'val': testloader}

#         model, info = train_model(model, dataloader,dataset_sizes, criterion, optimizer,scheduler,num_epochs=num_epochs)

#         #save the model onto disk
#         torch.save(model.state_dict(), base_PATH+ name)
#         #visualize the training and validation results
#         utils.show_info(info, 'ResNet152',figure_path+name)
#         testloader,dataset_sizes=CIFAR10(my_transforms.CIFAR10_transform(),my_transforms.CIFAR10_transform('val'),batch_size=batch_size,return_test_dataloader=True).return_dataloader()
#         test.test_model(model,testloader,dataset_sizes)

# if training==True:

#     #create a dataloader and choose a dataset
#     trainloader,testloader,dataset_sizes=CIFAR10(my_transforms.CIFAR10_transform(),my_transforms.CIFAR10_transform('val'),batch_size=batch_size).return_dataloader()
#     dataloader = {'train': trainloader, 'val': testloader}

#     # define the optimizer
#     criterion = nn.CrossEntropyLoss()

#     # define the loss function
#     optimizer = optim.Adam(model.parameters(), lr = lr)

#     ## Decay LR by a factor of 0.1 every 7 epochs
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=1)
#     #epochs
#     num_epochs=30
#     #train the model
#     model, info = train_model(model, dataloader,dataset_sizes, criterion, optimizer,scheduler,num_epochs=num_epochs)

#     #save the model onto disk
#     torch.save(model.state_dict(), PATH)
#     #visualize the training and validation results
#     utils.show_info(info, 'ViT base model')

# else:
    #create a dataloader and choose a dataset
    #if want to load model weights
