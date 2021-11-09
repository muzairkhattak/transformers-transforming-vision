#set the different parameters and model choice to train in this config file
import torch.nn as nn


class args:

    def __init__(self):
        #append the model names below which are to be trained:
        #only timm models are supported now
        self.model_list=['resnetv2_101x3_bitm','resnetv2_101x3_bitm_in21k']

        # define the optimizer
        self.criterion = nn.CrossEntropyLoss()

        #defiining hyperparameters #currently we are using cross-entropy loss for training
        self.batch_size = 512
        self.lr = 0.01

        #set path where you want to save model weights
        self.base_PATH='/home/uzair.khattak/ML_project/ml_project/saved_models/'

        #path to save train/test plots
        self.figure_path='/home/uzair.khattak/ML_project/ml_project/figures/'

        self.training=True  #if set to True, it will fine-tune the model using timm pretrained weights

        self.model_PATH= '/home/uzair.khattak/ML_project/ml_project/saved_models/recylce_material512by512'   #set path for already trained weights if only evaluating

        #set dataset to choose: can only set CIFAR-10 or CIFAR-100 for now

        self.dataset_name='CIFAR-10'
        self.classes=10


        #set the train validation split from the training set
        self.val_pct=0.1   
