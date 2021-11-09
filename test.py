#here we will write piece of code to just test the model on the test dataset



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
ImageFile.LOAD_TRUNCATED_IMAGES = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_model(model, dataloader,dataset_sizes):
    model=model.eval()
    model = model.to(device)
    since = time.time()
    # Iterate over data.
    running_corrects=0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward
        # track history if only in train
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / dataset_sizes
    time_elapsed = time.time() - since
    print('Inference completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Test accuracy: {:4f}'.format(epoch_acc))
