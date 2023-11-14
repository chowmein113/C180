import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
class TrainImageDataSet(Dataset):
    def __init__(self, image: np.array, num_sample: float):
        self.image = image
        self.num_sample = num_sample
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.coords = np.indices(self.image.shape[:2]).reshape(2, -1).T
    def __len__(self):
        return self.coords.shape[0]
    def __getitem__(self, idx):
        coord = self.coords[idx, ::-1] #(r, c) -> (x, y) format
        norm_coord = self.normalize_coords(coord)
        clr = self.image[coord[1], coord[0]] #back to coords
        return torch.from_numpy(norm_coord).float(), torch.from_numpy(clr).float()


class ImageDataSet(Dataset):
    def __init__(self, image: np.array, num_sample: float):
        """Ordered Image Dataset for prediction and test validation

        Args:
            img (np.array): rgb image as np array
            coords (np.array): N x 2 np array of coordinates in image
            num_sample (float): batch size
        """
        self.image = image
        self.num_sample = num_sample
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.coords = np.indices(self.image.shape[:2]).reshape(2, -1).T
        #get coords in N x 2 in (r, c) format
    def normalize_coords(self, coords: np.array) -> np.array:
        return coords / np.array([self.width - 1, self.height - 1])   
    def __len__(self):
        return self.coords.shape[0] // self.num_sample
    def __getitem__(self, idx):
        end = min((idx + 1) * self.num_sample, self.coords.shape[0])
        batch_coords = self.coords[idx * self.num_sample : end, ::-1] #(r, c) -> (x, y) format
        norm_coords = self.normalize_coords(batch_coords)
        clr = self.image[batch_coords[:, 1], batch_coords[:, 0]] #back to coords in (r, c) format to get pixels
        return torch.from_numpy(norm_coords).float(), torch.from_numpy(clr).float()
class RandomImageDataSet(ImageDataSet):
    def __init__(self, image: np.array, num_sample: float):
        super(RandomImageDataSet, self).__init__(image, num_sample)
        np.random.seed(42)
        np.random.shuffle(self.coords)
    def __len__(self):
        return self.coords.shape[0] // self.num_sample
    def __getitem__(self, idx):
        end = min((idx + 1) * self.num_sample, self.coords.shape[0])
        batch_coords = self.coords[idx * self.num_sample : end, ::-1] #(r, c) -> (x, y) format
        norm_coords = self.normalize_coords(batch_coords)
        clr = self.image[batch_coords[:, 1], batch_coords[:, 0]] #back to coords in (r, c) format to get pixels
        return torch.from_numpy(norm_coords).float(), torch.from_numpy(clr).float()
    # def __getitem__(self, _):
    #     img_coords = np.random.randint(0, [self.width, self.height], size = (self.num_sample, 2))
    #     norm_coords = self.normalize_coords(img_coords)
    #     m1 = np.min(norm_coords[:, 0])
    #     M1 = np.max(norm_coords[:, 0])
    #     m1 = np.min(norm_coords[:, 0])
    #     M1 = np.max(norm_coords[:, 0])
    #     clr = self.image[img_coords[:, 1], img_coords[:, 0]] / 255.0
    #     return torch.from_numpy(norm_coords).float(), torch.from_numpy(clr).float()
    # def __len__(self):
    #     return self.num_sample
    # def __getitem__(self, _):
    #     img_coord = np.random.randint(0, [self.width, self.height])
    #     norm_coord = self.normalize_coords(img_coord)
    #     clr = self.image[img_coord[1], img_coord[0]] / 255.0
    #     return torch.from_numpy(norm_coord).float(), torch.from_numpy(clr).float()
        
class ImageDataLoader(object):
    def __init__(self, sample_size: int = 10):
        self.sample_size = sample_size
        self.dataloaders = []
    
    
    def add_img(self, img: np.array):
        img_dataset = RandomImageDataSet(img, self.sample_size)
        img_loader = DataLoader(img_dataset, batch_size=1, shuffle=True)
        self.dataloaders.append(img_loader)
    def add_dataset(self, dataset: ImageDataSet):
        img_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.dataloaders.append(img_loader)
    def __len__(self):
        return len(self.dataloaders)
    def get_data_loader(self, idx):
        return self.dataloaders[idx]
    def add_data_set(self, dataset):
        self.dataloaders.append(dataset)