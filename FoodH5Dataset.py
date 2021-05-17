import ntpath
import torch
import h5py
import urllib
import numpy as np
from pathlib import Path

class FoodH5Dataset(torch.utils.data.Dataset):

    def __init__(self, root, download, url, transform=None):
        super(FoodH5Dataset, self).__init__()

        self.root = root
        self.url = url
        self.filename = ntpath.basename(url)
        self.filepath = f"{self.root}/{self.filename}"
        self.transform = transform

        if Path(self.filepath).is_file():
            if download:
                print(f"File {self.filepath} already exists")
        else:
            if download:
                self.download()
            else:
                print(f"File {self.filepath} does not exist, try setting download=True")

        h5_file = h5py.File(self.filepath, 'r')
        self.category = np.array(h5_file['category'], dtype=bool)
        self.category_names = np.array(h5_file['category_names'], dtype=str)
        self.images = np.array(h5_file['images'], dtype=np.float64)
        h5_file.close()


    def __getitem__(self, index):
        image = self.images[index, :, :, :]
        category_name = self.category_names[self.category[index]]
        category_idx = np.where(self.category_names == category_name)[0]

        X = torch.from_numpy(image).long()
        y = torch.from_numpy(category_idx).long()

        if self.transform:
            X = self.transform(X)

        return (X, y)
    
    def __len__(self):
        return self.images.shape[0]

    def download(self):
        print(f"Downloading from {self.url}")
        urllib.request.urlretrieve(self.url, self.filepath)
        print(f"{self.filepath} downloaded.")

