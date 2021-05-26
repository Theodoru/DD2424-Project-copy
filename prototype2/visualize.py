import os
import wide_resnet
import torchviz
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, Normalize

model = wide_resnet.Model(16, 4, 100)

test_transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize((0.486, 0.456, 0.406),
              (0.229, 0.224, 0.225)),
])

test_data = datasets.ImageFolder(root="data/images/test", transform=test_transform)
test_data_loader = DataLoader(test_data, batch_size=100,
                              shuffle=True, pin_memory=True, num_workers=os.cpu_count())
for X, y in test_data_loader:
    Y = model(X)
    torchviz.make_dot(Y.mean(), params=dict(model.named_parameters())).render(filename='model_viz')
    break
