import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize

# Configuration
BATCH_SIZE = 16
EPOCHS = 100
MODEL_ARCHITECTURE = 'wide_resnet50_2'
###############


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        print(f"train(): X type | dtype: {type(X)} | {X.dtype}, y type | dtype: {type(y)} | {y.dtype}")

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

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
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    # Create model state directory
    if not os.path.isdir("state"):
        os.mkdir("state")

    # Clear CUDA cache
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Datasets
    preprocess = Compose([
        # Resize(256),
        # CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # FIXME: Varför just dessa värden?????
    ])

    training_data = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=preprocess
    )
    test_data = datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=preprocess,
    )

    # Data loaders
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
