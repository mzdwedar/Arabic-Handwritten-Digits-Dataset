import torch.nn as nn
from torchvision.models import resnet18


def model_builder():
    """
    1. load resnet18 model
    2. change the number of out_features to the number of classes (i.e 10)
    3. change the number of input channels from 3 to 1

    returns: 
        restnet with 1 channel input and 10 classes output

    """
    model = resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, 10)

    return model

