from mlp import NeRF
# from scipy.ndimage import shift, map_coordinates
from tqdm import tqdm
# import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_loader import ImageDataSet, NerfDataSet

PART_1 = False
PART_2 = True
def load_img(path: str) -> np.array:
    img = skio.imread(path) / 255.0
    return img
def train(model: NeRF, num_iterations: int):
    data_loader = model.get_img_data_loader().get_data_loader(0)
    psnr = 0.0
    for i in tqdm(range(num_iterations), "Training Model..."):
        #batch
        for data in tqdm(data_loader, "batch..."):
            coords, goal_colors = data
            #flatten batch
            flattened_coords = coords.squeeze(0)
            flattened_goal_colors = goal_colors.squeeze(0)
            model.train(flattened_coords, flattened_goal_colors)
            psnr += model.get_psnrs()[-1]
            
        print(f"psnr final {i}: {psnr}")
        psnr = 0.0



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
    dataset = ImageDataSet(img, 10000)
    model.get_img_data_loader().add_dataset(dataset)
    for coords, goal_colors in tqdm(model.get_img_data_loader().get_data_loader(-1), "Testing..."):
        tcoords = coords.to(model.device)
        tcolors = goal_colors.to(model.device)
        flattened_coords = tcoords.squeeze(0)
        flattened_goal_colors = tcolors.squeeze(0)
        val = model.test(flattened_coords, flattened_goal_colors)
        coords = coords.numpy()[0] #coords in (x, y) format
        coords[:, 1] = coords[:, 1] * (height - 1)
        coords[:, 0] = coords[:, 0] * (width - 1)
        coords = np.round(coords).astype(int)
        # m1 = np.min(coords[:, 0])
        # M1 = np.max(coords[:, 0])
        # m2 = np.min(coords[:, 1])
        # M2 = np.max(coords[:, 1])
        canvas[coords[:, 1], coords[:, 0]] = val.cpu().float()
    # f, axs = plt.subplots(1,2)
    # f.suptitle("Where was canvas changed")
    # axs[0].imshow(canvas)
    # mask = (canvas != 0).astype(np.uint8)
    # m = np.max(mask)
    # axs[1].imshow(mask)
    # plt.show()
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
    
def part_1():
    ckpt = osp.join(osp.join(osp.abspath(osp.dirname(__file__)), "checkpoints", "nerf_complete.pth"))
    # nerf = NeRF(layers=6, learning_rate=1e-3, pth = ckpt)
    EPOCH = 1500
    LAYERS = 4
    LEARNING_RATE = 1e-3
    nerf = NeRF(layers=LAYERS, learning_rate=LEARNING_RATE)
    data_loader = nerf.get_img_data_loader()
    img_folder = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), "images")
    img1_pth = osp.join(img_folder, "beany.jpg")
    img1 = load_img(img1_pth)
    m=np.min(img1)
    M=np.max(img1)
    data_loader.add_img(img1)
    
    
    #train
    train(nerf, EPOCH)
    
    nerf.save_model(osp.join(osp.abspath(osp.dirname(__file__)), "checkpoints", f"mlp_epoch{EPOCH}_LR{LEARNING_RATE}_LAYER{LAYERS}.pth"))
    #metrics
    train_psnrs = nerf.get_psnrs()[:]
    nerf.psnrs = []
    pred = test(img1, nerf)
    psnrs = nerf.get_psnrs()
    # plt.imshow(img1)
    # plt.show()
    f, axs = plt.subplots(2, 2, figsize=(10, 10))
    f.suptitle(f"PNSRS and Image test w/ num_layers: {LAYERS}, " \
            + f"learning rate: {LEARNING_RATE}, epochs: {EPOCH}")
    axs[0, 0].plot(range(len(train_psnrs)), train_psnrs)
    axs[0, 0].set_title("PNSRS over train iterations")
    
    # pred = test(img1, nerf)
    # psnr = nerf.get_psnrs()
    # m = np.min(pred)
    # M = np.max(pred)
    axs[0, 1].imshow(np.round(pred * 255).astype(np.uint8))
    axs[0, 1].set_title("Regenerated image")
    axs[1, 0].imshow(img1)
    axs[1, 0].set_title("Original image")
    axs[1, 1].plot(range(len(psnrs)), psnrs)
    axs[1, 1].set_title("psnrs during test")
    plt.show()
    # plt.savefig(osp.join(img_folder, f"mlp_epoch{EPOCH}_LR{LEARNING_RATE}_LAYER{LAYERS}.png"))
#     return  
# def train_nerf(epoch: int, model: NeRF, c2ws_train, focal):
#     dataloader = model.get_img_data_loader().get_data_loader(-1)
    
#     for i in tqdm(range(epoch), "Training rays: "):
#         for batch in tqdm(dataloader, "Processing Batch: "):
            
        
def part_2():
    data_dir = osp.join(osp.abspath(osp.dirname(__file__)), "data")
    data = np.load(osp.join(data_dir, "lego_200x200.npz"))

    # Training images: [100, 200, 200, 3]
    images_train = data["images_train"] / 255.0

    # Cameras for the training images 
    # (camera-to-world transformation matrix): [100, 4, 4]
    c2ws_train = data["c2ws_train"]

    # Validation images: 
    images_val = data["images_val"] / 255.0

    # Cameras for the validation images: [10, 4, 4]
    # (camera-to-world transformation matrix): [10, 200, 200, 3]
    c2ws_val = data["c2ws_val"]

    # Test cameras for novel-view video rendering: 
    # (camera-to-world transformation matrix): [60, 4, 4]
    c2ws_test = data["c2ws_test"]

    # Camera focal length
    focal = data["focal"]  # float
    
    ckpt = osp.join(osp.join(osp.abspath(osp.dirname(__file__)), "checkpoints", "nerf_complete.pth"))
    # nerf = NeRF(layers=6, learning_rate=1e-3, pth = ckpt)
    EPOCH = 1500
    LAYERS = 4
    LEARNING_RATE = 1e-3
    nerf = NeRF(layers=LAYERS, learning_rate=LEARNING_RATE)
    dataset = NerfDataSet(images_train, 1000)
    nerf.get_img_data_loader().add_dataset(dataset)
    
    
    
    
    
def main():
    if PART_1:
        part_1()
    if PART_2:
        part_2()
    
    
if __name__ == "__main__":
    main()


