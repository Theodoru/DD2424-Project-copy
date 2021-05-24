import os
import json
import time
import datetime

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize
from torchvision.models.resnet import conv1x1, conv3x3
from typing import Callable, Optional

TEST_DIR = "/content/images/test"
TRAIN_DIR = "/content/images/train"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #Dropout?
        self.downsample = downsample
        self.stride = stride

        #self.dropout = nn.Dropout(0.35)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

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

        # Bias is False, because it is not needed when Conv2d followed by a BatchNorm, see:
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.bn = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(self.widths[0], group_blocks)
        self.layer2 = self._make_layer(self.widths[1], group_blocks, stride=2)
        self.layer3 = self._make_layer(self.widths[2], group_blocks, stride=2)

        self.bn = nn.BatchNorm2d(self.widths[2])
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(self.widths[2], num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1):
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                conv1x1(self.inplanes, planes, stride),
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

        x = self.maxpool(x)  # Why maxpool here?

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)  # Why ReLu here?

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
        optimizer.zero_grad(set_to_none=True)
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


DEFAULT_CONFIG = {
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "depth": 16,
    "width": 8,
    "dataset": "cifar100",
    "image_size": 32,
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
    if config["dataset"] == "food101":
        transform = Compose([
            Resize((config["image_size"], config["image_size"])),
            ToTensor(),
            Normalize((0.486, 0.456, 0.406),  ## Change normalization???
                      (0.229, 0.224, 0.225)),  ## Change normalization???
        ])

        training_data = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
        test_data = datasets.ImageFolder(root=TEST_DIR, transform=transform)
    else:
        preprocess = Compose([
            Resize(min(config["image_size"], 32)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    train_data_loader = DataLoader(training_data, batch_size=config["batch_size"],
                                   shuffle=True, pin_memory=True, num_workers=os.cpu_count())
    test_data_loader = DataLoader(test_data, batch_size=config["batch_size"],
                                  shuffle=True, pin_memory=True, num_workers=os.cpu_count())

    # Model
    # model = torch.hub.load('pytorch/vision:v0.9.0', MODEL_ARCHITECTURE, pretrained=False)
    model = Model(config["depth"], config["width"], num_classes=len(training_data.classes))
    model = model.to(device=DEVICE, memory_format=torch.channels_last)
    print(model)

    # Load previous model state if it exists
    state_path = "state/{dataset}_{size}x{size}_wide_{depth}_{width}"\
                 .format(dataset=config["dataset"], size=config["image_size"],
                         depth=config["depth"], width=config["width"])
    if os.path.exists(state_path):
        model.load_state_dict(torch.load(state_path))

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"], dampening=0)

    # PyTorch optimization
    torch.backends.cudnn.benchmark = True

    epoch_benchmark = []

    for t in range(config["epochs"]):
        print(f"Epoch {t+1}\n-------------------------------")
        start_time = time.perf_counter()
        train(train_data_loader, model, loss_fn, optimizer)
        torch.save(model.state_dict(), state_path)
        test(test_data_loader, model, loss_fn)
        end_time = time.perf_counter()

        runtime = end_time-start_time
        epoch_benchmark.append(runtime)
        print("Epoch runtime:", datetime.timedelta(seconds=runtime))
        print("Estimated finishing time:",
              datetime.datetime.now() + datetime.timedelta(seconds=np.average(epoch_benchmark)*(config["epochs"]-t-1)))
        print()


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
