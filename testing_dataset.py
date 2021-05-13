import torch
from FoodH5Dataset import FoodH5Dataset

train_dataset = FoodH5Dataset("./data/food_c101_n10099_r32x32x1.h5")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

for X, y in train_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
