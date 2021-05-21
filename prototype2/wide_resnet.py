import os
import json

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize
from torchvision.models.resnet import BasicBlock, conv1x1
from loaddata import create_dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self,
                 depth: int,
                 width: int,
                 num_classes: int):
        super(Model, self).__init__()

        # Code borrowed from:
        # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py
        self.depth = depth
        assert (self.depth - 4) % 6 == 0, 'depth should be 6n+4'
        group_blocks = (self.depth - 4) // 6
        self.widths = [int(v * width) for v in (16, 32, 64)]

        self.inplanes = 64  # filters
        #self.dilation = 1

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn = nn.BatchNorm2d(self.inplanes)

        # FIXME: Layers should be defined here:
        self.layer1 = self._make_layer(self.widths[0], group_blocks)
        self.layer2 = self._make_layer(self.widths[1], group_blocks, stride=2)
        self.layer3 = self._make_layer(self.widths[2], group_blocks, stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(self.widths[2], num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1):
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        else:
            downsample = None

        layers = []
        layers.append(BasicBlock(self.inplanes, planes=planes, downsample=downsample,
                                 stride=stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)  # What's the point of this first convolution?
        x = self.bn(x)  # Batch normalization
        x = F.relu(x, inplace=True)  # Why ReLu here?
        x = self.maxpool(x)  # Why maxpool here?

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x: (
        # BATCH_SIZE,
        # self.inplanes,
        # floor(((32 + 2*3 - 1*(7-1) - 1) / 2) + 1),
        # floor(((32 + 2*3 - 1*(7-1) - 1) / 2) + 1)
        # )
        x = self.avgpool(x)  # Why avgpool here?
        x = torch.flatten(x, 1)  # Puts all features in x in a vector.
        x = self.fc(x)  # Linear layer to acquire class-probabilities.

        return x


def train(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # print(f"train(): X type | dtype: {type(X)} | {X.dtype}, y type | dtype: {type(y)} | {y.dtype}")

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


DEFAULT_CONFIG = {
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "depth": 16,
    "width": 8,
    "dataset": "cifar100"
}


def load_config(config_path):
    if not os.path.exists(config_path):
        # Default values
        return DEFAULT_CONFIG

    with open(config_path, "r") as f:
        return {**DEFAULT_CONFIG, **json.load(f)}


def save_config(config_path, config):
    with open(config_path, "w") as f:
        return json.dump(config, f, indent=4, sort_keys=True)


def main(config):
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

    if config["dataset"] == "food101":
        training_data = create_dataset(True)
        test_data = create_dataset(False)
    else:
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
    train_data_loader = DataLoader(training_data, batch_size=config["batch_size"])
    test_data_loader = DataLoader(test_data, batch_size=config["batch_size"])

    # Model
    # model = torch.hub.load('pytorch/vision:v0.9.0', MODEL_ARCHITECTURE, pretrained=False)
    model = Model(config["depth"], config["width"], num_classes=len(training_data.classes))
    model.to(DEVICE)
    print(model)

    # Load previous model state if it exists
    #state_path = "state/" + MODEL_ARCHITECTURE
    #if os.path.exists(state_path):
    #    model.load_state_dict(torch.load(state_path))

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"])

    for t in range(config["epochs"]):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_data_loader, model, loss_fn, optimizer)
    #    torch.save(model.state_dict(), state_path)
        test(test_data_loader, model, loss_fn)


if __name__ == "__main__":
    # Load configuration
    config = load_config("config.json")

    try:
        main(config)
    except RuntimeError as error:
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        # Save configuration
        save_config("config.json", config)
        raise error
    except KeyboardInterrupt:
        print("Interrupted by user.")

    # Save configuration
    save_config("config.json", config)
