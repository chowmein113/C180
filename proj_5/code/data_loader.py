import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ImageDataLoader(object):
    def __init__(self, sample_size: int = 10):
        self.sample_size = sample_size
        self.dataloaders = []
    
    
    def add_img(self, img: np.array):
        img_dataset = ImageDataSet(img, self.sample_size)
        img_loader = DataLoader(img_dataset, batch_size=1, shuffle=True)
        self.dataloaders.append(img_loader)
    def __len__(self):
        return len(self.dataloaders)
    def get_data_loader(self, idx):
        return self.dataloaders[idx]
    
        
class ImageDataSet(Dataset):
    def __init__(self, image: np.array, num_sample: float):
        self.image = image
        self.num_sample = num_sample
        self.width = image.shape[1]
        self.height = image.shape[0]
    def __len__(self):
        return self.num_sample
    def __getitem__(self, _):
        img_coords = np.random.randint(0, [self.width, self.height], size = (len(self), 2))
        norm_coords = self.normalize_coords(img_coords)
        clr = self.image[img_coords[:, 1], img_coords[:, 0]] / 255.0
        return torch.from_numpy(norm_coords).float(), torch.from_numpy(clr).float()
        
    def normalize_coords(self, coords: np.array) -> np.array:
        return coords / np.array([self.width, self.height])