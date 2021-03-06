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
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize, RandomHorizontalFlip
from torchvision.models.resnet import conv1x1, conv3x3
from typing import Callable, Optional

import matplotlib.pyplot as plt

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
        self.dropout = nn.Dropout(0.4)
        self.conv2 = conv3x3(planes, planes)
        #Dropout?
        self.downsample = downsample
        self.stride = stride

        


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
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
    model.train()
    size = len(data_loader.dataset)
    losses = []
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
            losses.append(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return sum(losses) / len(losses)


def test(data_loader, model, loss_fn):
    model.eval()
    size = len(data_loader.dataset)
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
    return correct


DEFAULT_CONFIG = {
    "batch_size": 128,
    "epochs": 80,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "depth": 16,
    "width": 8,
    "dataset": "food101",
    "image_size": 128,
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


def load_epoch_runs(epoch_runs_path):
    if not os.path.exists(epoch_runs_path):
        # Default
        return []

    epoch_runs = []
    with open(epoch_runs_path, "r") as f:
        for line in f:
            epoch_runs.append(json.loads(line))

    return epoch_runs


def save_epoch_runs(epoch_runs_path, epoch_runs):
    with open(epoch_runs_path, "w") as f:
        for epoch_run in epoch_runs:
            f.write(json.dumps(epoch_run) + "\n")


def plot_loss(plot_path, epoch_runs, title=None):
    epoch_ids = [epoch_run["epoch_id"] for epoch_run in epoch_runs]
    losses = [epoch_run["avg_loss"] for epoch_run in epoch_runs]

    fig, ax = plt.subplots()
    ax.plot(epoch_ids, losses)

    ax.set(xlabel='Epoch', ylabel='Average loss', title=title if title else "Average loss during each epoch")
    ax.grid()

    fig.savefig(plot_path)
    plt.close()


def plot_runtime(plot_path, epoch_runs, title=None):
    epoch_ids = [epoch_run["epoch_id"] for epoch_run in epoch_runs]
    runtimes = [epoch_run["runtime"] for epoch_run in epoch_runs]

    fig, ax = plt.subplots()
    ax.plot(epoch_ids, runtimes)

    ax.set(xlabel='Epoch', ylabel='Runtime (s)', title=title if title else "Runtime for each epoch")
    ax.grid()

    fig.savefig(plot_path)
    plt.close()

def plot_test_accuracy(plot_path, epoch_runs, title=None):
    epoch_ids = [epoch_run["epoch_id"] for epoch_run in epoch_runs]
    accuracies = [epoch_run["test_accuracy"] for epoch_run in epoch_runs]

    fig, ax = plt.subplots()
    ax.plot(epoch_ids, accuracies)

    ax.set(xlabel='Epoch', ylabel='Test Accuracy', title=title if title else "Test accuracy development")
    ax.grid()

    fig.savefig(plot_path)
    plt.close()

def main(config):
    # Create model state directories
    if not os.path.isdir("state"):
        os.mkdir("state")

    # Clear CUDA cache
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Datasets
    if config["dataset"] == "food101":
        testtransform = Compose([
            Resize((config["image_size"], config["image_size"])),
            ToTensor(),
            Normalize((0.486, 0.456, 0.406),  ## Change normalization???
                      (0.229, 0.224, 0.225)),  ## Change normalization???
        ])
        traintransform = Compose([
            #RandomHorizontalFlip(),
            testtransform,
        ])

        training_data = datasets.ImageFolder(root=TRAIN_DIR, transform=traintransform)
        test_data = datasets.ImageFolder(root=TEST_DIR, transform=testtransform)
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
        if os.path.exists(state_path + "/last_model"):
            model.load_state_dict(torch.load(state_path + "/last_model"))

        if os.path.exists(state_path + "/best_model"):
            print('loaded best')
            model.load_state_dict(torch.load(state_path + "/best_model"))

        epoch_runs = load_epoch_runs(state_path + "/epoch_runs.jl")
    else:
        os.mkdir(state_path)
        epoch_runs = []

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"],
                                weight_decay=config["weight_decay"], dampening=0)

    #schduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,60], gamma=0.1)

    # PyTorch optimization
    torch.backends.cudnn.benchmark = True

    epoch_benchmark = []

    with open(state_path + "/epoch_runs.jl", "a") as f:
        best_accuracy = max((epoch_run["test_accuracy"] for epoch_run in epoch_runs), default=0)
        first_id = max((epoch_run["epoch_id"] for epoch_run in epoch_runs), default=0) + 1

        for t, epoch_id in enumerate(range(first_id, first_id + config["epochs"])):
            print(f"Epoch {epoch_id}\n-------------------------------")
            start_time = time.perf_counter()
            print(scheduler.get_last_lr())
            avg_loss = train(train_data_loader, model, loss_fn, optimizer)
            torch.save(model.state_dict(), state_path + "/last_model")
            accuracy = test(test_data_loader, model, loss_fn)

            scheduler.step()

            if accuracy > best_accuracy:
                torch.save(model.state_dict(), state_path + "/best_model")

            end_time = time.perf_counter()
            runtime = end_time-start_time

            epoch_run = {"epoch_id": epoch_id, "avg_loss": avg_loss, "test_accuracy": accuracy,
                         "runtime": runtime, "config": config}
            epoch_runs.append(epoch_run)
            f.write(json.dumps(epoch_run) + "\n")
            f.flush()

            # Plot
            plot_loss(state_path + "/plot_avg_loss.png", epoch_runs)
            plot_runtime(state_path + "/plot_runtime.png", epoch_runs)
            plot_test_accuracy(state_path + "/plot_test_accuracy.png", epoch_runs)

            epoch_benchmark.append(runtime)
            print("Epoch runtime:", datetime.timedelta(seconds=runtime))
            print("Estimated finishing time:",
                  datetime.datetime.now() +
                  datetime.timedelta(seconds=np.average(epoch_benchmark)*(config["epochs"]-t-1)))
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
