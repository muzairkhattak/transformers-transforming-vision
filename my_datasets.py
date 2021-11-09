import torch
import torchvision
from torch.utils.data import random_split #issue in this is that both validation and training will get the same transforms
from torch.utils.data.sampler import SubsetRandomSampler  #here by using this, we can deal with validation and train set individually wrt transforms
import numpy as np
class CIFAR10:

    def __init__(self,transforms_train,transforms_val,batch_size,return_test_dataloader=False,val_pct=0.02):
        self.transforms_train=transforms_train
        self.transforms_test=transforms_val
        self.batch_size=batch_size
        self.val_pct=val_pct
        self.return_test_dataloader=return_test_dataloader

    def return_dataloader(self):
        #here we will split the original train loader into trainset and the validation set
                # load the dataset
        if self.return_test_dataloader==False: #this means to return the training and validation dataset
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transforms_train)

            valid_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transforms_test)

            num_train = len(train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.val_pct * num_train))
            #also lets shuffle out dataset images
            np.random.shuffle(indices)

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                            num_workers=2)
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                num_workers=2)

            dataset_sizes = {'train':len(train_loader)*self.batch_size,'val':len(valid_loader)*self.batch_size}

            return train_loader, valid_loader,dataset_sizes
        else:  #return the test loader
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transforms_test)

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size,
                num_workers=2)
            dataset_sizes_test=len(test_loader)*self.batch_size
            
            return test_loader,dataset_sizes_test




class CIFAR100:

    def __init__(self,transforms_train,transforms_val,batch_size,return_test_dataloader=False,val_pct=0.02):
        self.transforms_train=transforms_train
        self.transforms_test=transforms_val
        self.batch_size=batch_size
        self.val_pct=val_pct
        self.return_test_dataloader=return_test_dataloader
    def return_dataloader(self):
        #here we will split the original train loader into trainset and the validation set
                # load the dataset
        if self.return_test_dataloader==False: #this means to return the training and validation dataset
            train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transforms_train)

            valid_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=self.transforms_test)

            num_train = len(train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(self.val_pct * num_train))
            #also lets shuffle out dataset images
            np.random.shuffle(indices)

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                            num_workers=2)
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                num_workers=2)

            dataset_sizes = {'train':len(train_loader)*self.batch_size,'val':len(valid_loader)*self.batch_size}

            return train_loader, valid_loader,dataset_sizes
        else:  #return the test loader
            test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=self.transforms_test)

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size,
                num_workers=2)
            dataset_sizes_test=len(test_loader)*self.batch_size

            return test_loader,dataset_sizes_test
