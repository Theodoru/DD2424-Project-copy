import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import numpy as np

# Add parent dir to sys.path so we can import if we are in prototype folder
sys.path[0] += "/.."

from loaddata import create_dataset

IMAGE_SHAPE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 5
MODEL_ARCHITECTURE = "wide_resnet50_2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR = "../../archive.nosync/images/test"
TRAIN_DIR = "../../archive.nosync/images/train"
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
epoch_step = [60,120,160]
lr_decay_ratio = 0.2

def train(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        #print(X.dtype)
        #print(y.dtype)

        # Compute prediction error
        pred = model(X)
        #print(f"what is this?: {y}, {pred}")
        loss = loss_fn(pred, y.flatten())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            print(f"what is this?: {y}")
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    # Needs to be run from root directory
    print(f"Current directory: {os.getcwd()}")

    # Create model state directory
    if not os.path.isdir("state"):
        os.mkdir("state")

    # Clear CUDA cache
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Load datasets
    training_data = create_dataset(True, IMAGE_SHAPE, TEST_DIR)
    test_data = create_dataset(True, IMAGE_SHAPE, TRAIN_DIR)

    # Setup data loaders
    train_data_loader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Model
    model = torch.hub.load('pytorch/vision:v0.9.0', MODEL_ARCHITECTURE, pretrained=False)
    model.to(DEVICE)
    print(model)

    # Load previous model state if it exists
    state_path = "state/" + MODEL_ARCHITECTURE
    if os.path.exists(state_path):
        model.load_state_dict(torch.load(state_path))

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    for t in range(EPOCHS):
        if t in (epoch_step):
            lr = LR * lr_decay_ratio
            optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

        print(f"Epoch {t+1}\n-------------------------------")
        train(train_data_loader, model, loss_fn, optimizer)
        torch.save(model.state_dict(), state_path)
        test(test_data_loader, model, loss_fn)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        raise error
    except KeyboardInterrupt:
        print("Interrupted by user.")
