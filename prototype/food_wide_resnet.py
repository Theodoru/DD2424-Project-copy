
import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize

# Add parent dir to sys.path so we can import if we are in prototype folder
sys.path[0] += "/.."
from FoodH5Dataset import FoodH5Dataset

BATCH_SIZE = 16
EPOCHS = 5
MODEL_ARCHITECTURE = "wide_resnet50_2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
S3_URL = "https://dd2424-project.s3.eu-north-1.amazonaws.com"
TRAINING_DATA = "food_c101_n10099_r32x32x3.h5"
TEST_DATA = "food_test_c101_n1000_r32x32x3.h5"


def train(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        print(X.dtype)
        print(y.dtype)

        # Compute prediction error
        pred = model(X)
        print(f"what is this?: {y}, {pred}")
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


def preprocess(X):
    X = torch.reshape(X, [3, 32, 32])
    X = X.type(torch.LongTensor)
    return X 


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
    training_data = FoodH5Dataset(root="data", download=True, url=f"{S3_URL}/{TRAINING_DATA}", transform=preprocess)
    test_data = FoodH5Dataset(root="data", download=True, url=f"{S3_URL}/{TEST_DATA}", transform=preprocess)

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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(EPOCHS):
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
