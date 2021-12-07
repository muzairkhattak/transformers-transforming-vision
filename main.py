#importing basic libraries
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import ImageFile
# from efficientnet_pytorch import EfficientNet
ImageFile.LOAD_TRUNCATED_IMAGES = True
#custom libraries
from models import Net,VisionTransformers,ResNet152,BiT
from train import train_model
import my_transforms
from my_datasets import CIFAR10, CIFAR100,CUB_dataloader
import utils
import test
import argparse

def get_args_parser():

    parser = argparse.ArgumentParser('ML701 Project- Training and Evaluating PyTorch timm models', add_help=False)

    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--num-epochs', default=1, type=int)

    # Model parameters
    parser.add_argument('--model-name', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--img-size', default=224, type=int, help='input image size/resolution')

    # Learning rate schedule parameters
    parser.add_argument('--optimizer', default='ADAM',choices=['ADAM', 'SGD'], type=str,
                        help='Optimizer for training (default: ADAM)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    # * Finetuning params
    parser.add_argument('--training', default=True, help='finetune from checkpoint (default: True), put False for evaluation')

    parser.add_argument('--weights-path', default='./saved_models/', type=str,
                        help='path where you want to save model weights after training')

    parser.add_argument('--load-weights', default='./saved_models/', type=str,
                        help='path from where trained weights are to be loaded')

    parser.add_argument('--figure-path', default='./figures/', type=str,
                        help='path for saving plots')

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/', type=str,
                        help='path where dataset is present')
    parser.add_argument('--dataset-name', default='CUB200', choices=['CIFAR10', 'CIFAR100', 'CUB200'],
                        type=str, help='Choice of dataset, default= CUB200')

    parser.add_argument('--val-pct', default=0.1, type=float,
                        help='validation split from the training set (default:0.1')


    return parser



def main(Args):


    print(Args)
    models_list=['resnetv2_101x3_bitm','resnetv2_101x3_bitm_in21k']
        #create a dataloader and choose a dataset
    Args.img_size=(Args.img_size,Args.img_size)
    if Args.dataset_name=='CIFAR10':
        trainloader,val_loader,dataset_sizes_train=CIFAR10(my_transforms.CIFAR10_transform(img_size=Args.img_size),my_transforms.CIFAR10_transform('val',img_size=Args.img_size),batch_size=Args.batch_size,val_pct=Args.val_pct).return_dataloader()
        dataloader = {'train': trainloader, 'val': val_loader}
        testloader,dataset_sizes_test=CIFAR10(my_transforms.CIFAR10_transform(img_size=Args.img_size),my_transforms.CIFAR10_transform('val'),batch_size=Args.batch_size,return_test_dataloader=True).return_dataloader()
        classes=10
        

    elif Args.dataset_name=='CIFAR100':

        trainloader,val_loader,dataset_sizes_train=CIFAR100(my_transforms.CIFAR100_transform(img_size=Args.img_size),my_transforms.CIFAR100_transform('val',img_size=Args.img_size),batch_size=Args.batch_size,val_pct=Args.val_pct).return_dataloader()
        dataloader = {'train': trainloader, 'val': val_loader}
        testloader,dataset_sizes_test=CIFAR100(my_transforms.CIFAR100_transform(img_size=Args.img_size),my_transforms.CIFAR100_transform('val',img_size=Args.img_size),batch_size=Args.batch_size,return_test_dataloader=True).return_dataloader()
        classes=100

    elif Args.dataset_name=='CUB200':

        trainloader,val_loader,dataset_sizes_train=CUB_dataloader(my_transforms.CUB_transform(img_size=Args.img_size),my_transforms.CUB_transform('val',img_size=Args.img_size),batch_size=Args.batch_size,val_pct=Args.val_pct).return_dataloader()
        dataloader = {'train': trainloader, 'val': val_loader}
        testloader,dataset_sizes_test=CUB_dataloader(my_transforms.CUB_transform(img_size=Args.img_size),my_transforms.CUB_transform('val',img_size=Args.img_size),batch_size=Args.batch_size,return_test_dataloader=True).return_dataloader()
        classes=200


    if Args.training==True:

        criterion= nn.CrossEntropyLoss()
        #loading the relevent models
        if Args.model_name in models_list:
            model=BiT(classes,Args.model_name).return_model()
        else:
            model=VisionTransformers(classes,Args.model_name,img_size=Args.img_size).return_model()

        # define the loss function
        if Args.optimizer=='ADAM':
            optimizer = optim.Adam(model.parameters(), lr = Args.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr = Args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.05)
        print("Training model : ", Args.model_name)
        name=Args.model_name+ "_" + str(Args.num_epochs)+ "_" + str(Args.batch_size)+ Args.dataset_name

        model, info = train_model(model,Args.model_name, dataloader,dataset_sizes_train,criterion, optimizer,scheduler,num_epochs=Args.num_epochs)


        #save the model onto disk
        torch.save(model.state_dict(), Args.weights_path+ name)
        #visualize the training and validation results
        utils.show_info(info, Args.model_name,Args.figure_path+name)
        utils.show_info_loss(info, Args.model_name,Args.figure_path+name)
        test.test_model(model,testloader,dataset_sizes_test)


    else:  #just load the trained weights and evaluate on the test dataset


        if Args.model_name in models_list:
            model=BiT(classes,Args.model_name).return_model()
        else:
            model=VisionTransformers(classes,Args.model_name,Args.img_size).return_model()

        print("Loading weights from provided path....")
        model.load_state_dict(torch.load(Args.load_weights))

        print("Evaluating model : ", Args.model_name)
        test.test_model(model,testloader,dataset_sizes_test)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('ML701 Project- Training and Evaluating PyTorch timm models', parents=[get_args_parser()])
    Args = parser.parse_args()
    main(Args)