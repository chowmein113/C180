from mlp import NeRF
from scipy.ndimage import shift, map_coordinates
import skimage.io as skio
import skimage.transform as skt
from skimage.draw import polygon
from skimage import color
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_loader import ImageDataSet
def load_img(path: str) -> np.array:
    img = skio.imread(path) / 255.0
    return img
def train(model: NeRF, num_iterations: int):
    data_loader = model.get_img_data_loader().get_data_loader(0)
    for _ in tqdm(range(num_iterations), "Training Model..."):
        #batch
        for idx, data in tqdm(enumerate(data_loader), "batch..."):
            coords, goal_colors = data
            #flatten batch
            flattened_coords = coords.squeeze(0)
            flattened_goal_colors = goal_colors.squeeze(0)
            model.train(flattened_coords, flattened_goal_colors)
@torch.no_grad()           
def pred_img(img, model):
    height, width = img.shape[:2]
    
    canvas = np.zeros(img.shape)
    #corners in r, c
    pts = np.array([[0, 0],
                    [height - 1, 0],
                    [0, width - 1],
                    [height - 1, width - 1]])
    r, c = polygon(pts[:, 0], pts[:, 1])
    coords = np.array([r, c]).T
    canvas[coords[:, 0], coords[:, 1]] = model.pred(coords)
    return canvas
@torch.no_grad()
def test(img, model: NeRF):
    
    height, width = img.shape[:2]
    
    canvas = np.zeros(img.shape)
    # mask = np.zeros(img.shape)
    #corners in r, c
    # dataset = ImageDataSet(img, coords)
    # model.get_img_data_loader().add_data_set(dataset)
    # for i in tqdm(range(coords.shape[0]), "predicting"):
    #     tcoords = torch.from_numpy(coords[i, :]).to(model.device).float()
    #     tcolors = torch.from_numpy(img[coords[i, 0], coords[i, 1]]).to(model.device).float()
    #     flattened_coords = tcoords.squeeze(0)
    #     flattened_goal_colors = tcolors.squeeze(0)
    #     val = model.test(flattened_coords, flattened_goal_colors)
    #     canvas[coords[i, 0], coords[i, 1]] = val.cpu().float()
    dataset = ImageDataSet(img, height * width)
    model.get_img_data_loader().add_dataset(dataset)
    for coords, goal_colors in tqdm(model.get_img_data_loader().get_data_loader(-1), "Testing..."):
        tcoords = coords.to(model.device)
        tcolors = goal_colors.to(model.device)
        flattened_coords = tcoords.squeeze(0)
        flattened_goal_colors = tcolors.squeeze(0)
        val = model.test(flattened_coords, flattened_goal_colors)
        coords = coords.numpy()[0]
        coords[:, 0] = coords[:, 0] * height
        coords[:, 1] = coords[:, 1] * width
        coords = np.round(coords).astype(int)
        m1 = np.min(coords[:, 0])
        M1 = np.max(coords[:, 0])
        m2 = np.min(coords[:, 1])
        M2 = np.max(coords[:, 1])
        canvas[coords[:, 0], coords[:, 1]] = val.cpu().float()
    f, axs = plt.subplots(1,2)
    f.suptitle("Where was canvas changed")
    axs[0].imshow(canvas)
    mask = (canvas != 0).astype(np.uint8)
    m = np.max(mask)
    axs[1].imshow(mask)
    plt.show()
    return canvas 
    
def model_process(data_queue, result_queue, model: NeRF):
    
    while True:
        data = data_queue.get()
        if data is None:
            break
        with torch.no_grad(): 
            input_parm = data.to(model.device)
            result = model.pred(input_parm)
            result_queue.put(result.cpu()) 
    
    
def main():
    nerf = NeRF()
    data_loader = nerf.get_img_data_loader()
    img_folder = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), "images")
    img1_pth = osp.join(img_folder, "beany.jpg")
    img1 = load_img(img1_pth)
    data_loader.add_img(img1)
    
    
    #train
    train(nerf, 3000)
    
    nerf.save_model(osp.join(osp.abspath(osp.dirname(__file__)), "checkpoints", "nerf.pth"))
    #metrics
    pred = test(img1, nerf)
    psnrs = nerf.get_psnrs()
    plt.imshow(img1)
    plt.show()
    f, axs = plt.subplots(1, 2, figsize=(10, 10))
    f.suptitle("PNSRS and Image test")
    axs[0].plot(range(len(psnrs)), psnrs)
    axs[0].set_title("PNSRS over train iterations")
    
    pred = test(img1, nerf)
    # psnr = nerf.get_psnrs()
    m = np.min(pred)
    M = np.max(pred)
    axs[1].imshow(np.round(pred * 255).astype(np.uint8))
    axs[1].set_title("Regenerated image")
    plt.show()
    return
if __name__ == "__main__":
    main()


