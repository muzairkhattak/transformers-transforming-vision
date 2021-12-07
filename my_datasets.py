import torch
import torchvision
from torch.utils.data import random_split #issue in this is that both validation and training will get the same transforms
from torch.utils.data.sampler import SubsetRandomSampler  #here by using this, we can deal with validation and train set individually wrt transforms
import numpy as np
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset


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
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transforms_test)

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size,
                num_workers=2)
            dataset_sizes_test=len(test_loader)*self.batch_size

            return test_loader,dataset_sizes_test


class CUB(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root='./data', train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target



class CUB_dataloader:

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
            train_dataset=CUB(transform=self.transforms_train)

            valid_dataset=CUB(transform=self.transforms_test)


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
            test_dataset = CUB(transform=self.transforms_test,train=False)

            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size,
                num_workers=2)
            dataset_sizes_test=len(test_loader)*self.batch_size

            return test_loader,dataset_sizes_test