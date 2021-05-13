import torch
import h5py
import numpy as np


class FoodH5Dataset(torch.utils.data.Dataset):

    def __init__(self, file_path):
        super(FoodH5Dataset, self).__init__()
        h5_file = h5py.File(file_path, 'r')
        self.category = np.array(h5_file['category'], dtype=bool)
        self.category_names = np.array(h5_file['category_names'], dtype=str)
        self.images = np.array(h5_file['images'], dtype=np.float64)
        h5_file.close()


    def __getitem__(self, index):
        image = self.images[index, :, :, :]
        category_name = self.category_names[self.category[index]]
        category_idx = np.where(self.category_names == category_name)[0]
        return (torch.from_numpy(image), torch.from_numpy(category_idx))
    
    def __len__(self):
        return self.images.shape[0]

