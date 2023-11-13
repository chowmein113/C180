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
from multiprocessing import Pool, cpu_count
def load_img(path: str) -> np.array:
    img = skio.imread(path) / 255.0
    return img
def train(model: NeRF, num_iterations: int):
    data_loader = model.get_img_data_loader().get_data_loader(0)
    for _ in tqdm(range(num_iterations), "Training Model..."):
        #batch
        for coords, goal_colors in tqdm(data_loader, "Processing Batch"):
            #flatten batch
            flattened_coords = coords.squeeze(0)
            flattened_goal_colors = goal_colors.squeeze(0)
            model.train(flattened_coords, flattened_goal_colors)
            
def pred_img(img, model):
    height, width = img.shape[:2]
    
    canvas = np.zeros(img)
    #corners in r, c
    pts = np.array([[0, 0],
                    [height - 1, 0],
                    [0, width - 1],
                    [height - 1, width - 1]])
    r, c = polygon(pts[:, 0], pts[:, 1])
    coords = np.array([r, c]).T
    canvas[coords[:, 0], coords[:, 1]] = model.pred(coords)
    return canvas
def test(img, model: NeRF):
    height, width = img.shape[:2]
    
    canvas = np.zeros(img)
    #corners in r, c
    pts = np.array([[0, 0],
                    [height - 1, 0],
                    [0, width - 1],
                    [height - 1, width - 1]])
    r, c = polygon(pts[:, 0], pts[:, 1])
    coords = np.array([r, c]).T
    canvas[coords[:, 0], coords[:, 1]] = model.test(coords, img[coords[:, 0], coords[:, 1]])
    return canvas
    
    
    
    
def main():
    nerf = NeRF()
    data_loader = nerf.get_img_data_loader()
    img_folder = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), "images")
    img1_pth = osp.join(img_folder, "beany.jpg")
    img1 = load_img(img1_pth)
    data_loader.add_img(img1)
    
    
    #train
    train(nerf, 1500)
    
    nerf.save_model(osp.join(osp.abspath(osp.dirname(__file__)), "checkpoints", "nerf.pth"))
    #metrics
    pred = test(img1, nerf)
    psnrs = nerf.get_psnrs()
    
    f, axs = plt.subplots(1, 2, figsize=(10, 10))
    f.suptitle("PNSRS and Image test")
    axs[0].plot(range(len(psnrs)), psnrs)
    axs[0].set_title("PNSRS over train iterations")
    
    pred = test(img1, nerf)
    # psnr = nerf.get_psnrs()
    axs[1].imshow(pred)
    axs[1].set_title("Regenerated image")
    plt.show()
    
if __name__ == "__main__":
    main()


