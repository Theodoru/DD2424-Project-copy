import torch
import torchvision

import torchvision.transforms as T
from torch.utils.data import DataLoader

import numpy as np


IMG_SHAPE = (32,32)
TEST_DIR = "../archive.nosync/images/test"
TRAIN_DIR = "../archive.nosync/images/train"

##Image preprocessing from CIFAR in paper
def create_dataset(train):
    transform = T.Compose([
        T.Resize(IMG_SHAPE),
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, ## Change normalization???
                    np.array([63.0, 62.1, 66.7]) / 255.0), ## Change normalization???
    ])
    if train:
      transform = T.Compose([
        T.Pad(4, padding_mode='reflect'), ##Change value to funtion of size
        T.RandomHorizontalFlip(),
        T.RandomCrop(32), ## Change value to size???
        transform
        ])

      return torchvision.datasets.ImageFolder(root=TRAIN_DIR,
                                              transform=transform
)
    else:
      return torchvision.datasets.ImageFolder(root=TEST_DIR, 
                                              transform=transform)

test_ds = create_dataset(False)
train_ds = create_dataset(True)

print("test images = ", test_ds.idx_to_class)
#print("train images = ", train_ds)