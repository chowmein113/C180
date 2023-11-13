import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from data_loader import ImageDataSet, ImageDataLoader
import os.path as osp

def positional_encoding(x, L=10):
    pe = [x]
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn(2. ** i * x))
    return torch.cat(pe, dim=-1)

class MLP(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_size: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #hyper parameters
        self.L = 10
        input_dims = 2 * (2 * self.L + 1)
        layers = [nn.Linear(input_dims, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.extend([nn.Linear(hidden_size, 3), nn.Sigmoid()])  # Output RGB
    
        self.layers = nn.Sequential(*layers)

    def forward(self, coords):
        pe_coords = positional_encoding(coords, L=self.L)
        return self.layers(pe_coords)
class NeRF(object):
    def __init__(self, gpu_id: int = 0, pth: str = ""):
        
        
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = MLP()
    
        #load model weights/bias if exists
        if pth != "":
            self.model.load_state_dict(torch.load(pth))
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)
        self.ImageDataLoader = ImageDataLoader(sample_size=10000)
        self.psnrs = np.array([])
        self.model.to(device)
    
    def get_psnrs(self):
        return self.psnrs
    def psnr(self, mse: torch.Tensor, max_pixel_val: float = 1.0) -> torch.Tensor:
        inside = torch.tensor((max_pixel_val ** 2) / mse).to(self.device)
        return 10 * torch.log10(inside)
    def get_img_data_loader(self) -> ImageDataLoader:
        return self.ImageDataLoader
    def save_model(self, pth: str = osp.join(osp.dirname(osp.abspath(__file__)), "checkpoints", "nerf.pth")):
        #save model weights
        torch.save(self.model.state_dict(), pth)
        return
    def train(self, coords, actual_colors):
        #forward
        coords = coords.to(self.device)
        actual_colors = actual_colors.to(self.device)
        self.optimizer.zero_grad()
        pred = self.pred(coords)
        loss = self.criterion(pred, actual_colors)
        
        #back propagation
        loss.backward()
        self.optimizer.step()
        
        #Calculate loss
        mse = loss.item()
        
        return
    def pred(self, coords):
        predict = self.model(coords)
        return predict
        
    def test(self, coords, actual_colors):
        pred = self.pred(coords)
        loss = self.criterion(pred, actual_colors)
        mse = loss.item()
        np.append(self.psnrs, self.psnr(mse))
        return pred
          
        
    
    


