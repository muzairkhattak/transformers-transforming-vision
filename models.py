import torch 
import torch.nn as nn
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet

from torchvision import models
import timm

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet152:
    def __init__(self,class_num,model_version):
        super().__init__()
        self.model_ft = timm.create_model(model_version,pretrained = True)  #we will do the finetuning for now
        self.classes=class_num
    
    def return_model(self):
        num_ftrs = self.model_ft.fc.in_features
        #now also, for finetuning- please freeze all the initial layers except the last linear layer
        for param in self.model_ft.parameters():
            param.requires_grad = False
        self.model_ft.fc = nn.Linear(num_ftrs,self.classes)

        return self.model_ft


# class EfficientNetL2:
#     def __init__(self, class_num):
#         super().__init__()
#         self.model_ft = EfficientNet.from_pretrained('efficientnet-b7')
#         self.num_ftrs = model_ft.fc.in_features
#                 self.classes=class_num

#         model_ft.fc = nn.Linear(num_ftrs,class_num)
    
#     def return_model(self):
#         return self.model_ft

#both ViT and DeiT
class VisionTransformers:
    def __init__(self, class_num,model_version):
        super().__init__()
        self.model_version=model_version
        self.model_ft = timm.create_model(model_version,pretrained = True)  #we will do the finetuning for now
        self.classes=class_num


    def return_model(self):
        head_input_features=self.model_ft.head.in_features
        #now also, for finetuning- please freeze all the initial layers except the last linear layer
        for param in self.model_ft.parameters():
            param.requires_grad = False

        if self.model_version=='deit_base_distilled_patch16_224':
            self.model_ft.head = nn.Linear(head_input_features,self.classes)
            self.model_ft.head_dist = nn.Linear(head_input_features,self.classes)
        else:
            #for ViT
            self.model_ft.head = nn.Linear(head_input_features,self.classes)
        return self.model_ft
  
class BiT:
    def __init__(self, class_num,model_version):
        super().__init__()
        self.model_version=model_version
        self.model_ft = timm.create_model(model_version,pretrained = True)  #we will do the finetuning for now
        self.classes=class_num


    def return_model(self):
        head_input_features=self.model_ft.head.fc.in_channels
        #now also, for finetuning- please freeze all the initial layers except the last linear layer
        for param in self.model_ft.parameters():
            param.requires_grad = False

        self.model_ft.head.fc = nn.Conv2d(head_input_features,self.classes,kernel_size=(1, 1), stride=(1, 1))
        return self.model_ft





















    