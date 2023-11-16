from mlp import NeRF, DeepNeRF
# from scipy.ndimage import shift, map_coordinates
from tqdm import tqdm
# import os
import os.path as osp
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
from data_loader import ImageDataSet, NerfDataSet, NerfSingularDataSet
import skimage.io as skio
from tools import *
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
def calibrate_part1():
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
    dataset = NerfSingularDataSet(data=images_train, num_samples=1000, num_workers = 4, f = focal, c2w = c2ws_train, im_height=200, im_width=200)
    # dataloader = nerf.get_img_data_loader().add_dataset(dataset)
    import viser, time  # pip install viser
    # import numpy as np

    # --- You Need to Implement These ------
    # dataset = RaysData(images_train, K, c2ws_train)
    ray_color_tensors = dataset.sample_rays(100)
    # rays_o, rays_d, pixels = dataset.sample_rays(100) # Should expect (B, 3)
    rays_o = ray_color_tensors[:, 1:, 0].numpy()
    rays_d = ray_color_tensors[:, 1:, 1].numpy()
    
    
    points = sample_along_rays(ray_color_tensors, near=2.0, far=6.0, samples=32, perturb=True)
    H, W = images_train.shape[1:3]
    K = intrinsic_K(focal, H, W)
    # ---------------------------------------

    server = viser.ViserServer(share=True)
    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image
        )
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        server.add_spline_catmull_rom(
            f"/rays/{i}", positions=np.stack((o, o + d * 6.0)),
        )
    server.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.02,
    )
    time.sleep(1000)  
def calibrate_part2():   
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
    dataset = NerfSingularDataSet(data=images_train, num_samples=1000, num_workers = 4, f = focal, c2w = c2ws_train, im_height=200, im_width=200)  
    # Visualize Cameras, Rays and Samples
    import viser, time
    # import numpy as np

    # --- You Need to Implement These ------
    # dataset = RaysData(images_train, K, c2ws_train)

    
    
    # # Uncoment this to display random rays from the first image
    indices = np.random.randint(low=0, high=40_000, size=100)

    # # Uncomment this to display random rays from the top left corner of the image
    # indices_x = np.random.randint(low=100, high=200, size=100)
    # indices_y = np.random.randint(low=0, high=100, size=100)
    # indices = indices_x + (indices_y * 200)
    
    ray_color_tensors = dataset.get_rays_by_idx(indices)
    # rays_o, rays_d, pixels = dataset.sample_rays(100) # Should expect (B, 3)
    rays_o = ray_color_tensors[:, 1:, 0].numpy()
    rays_d = ray_color_tensors[:, 1:, 1].numpy()
    pixels = ray_color_tensors[:, 1:, 2].numpy()
    
    # This will check that your uvs aren't flipped
    uvs_start = 0
    uvs_end = 40_000
    assert dataset.camera_pixel_pairs is not None
    # sample_uvs = np.round((dataset.camera_pixel_pairs[uvs_start:uvs_end, 1:3] - 0.5).numpy()).astype(int) # These are integer coordinates of widths / heights (xy not yx) of all the pixels in an image
    # uvs are array of xy coordinates, so we need to index into the 0th image tensor with [0, height, width], so we need to index with uv[:,1] and then uv[:,0]
    # pixels = dataset.camera_pixel_pairs[uvs_start:uvs_end, 3:]
    # for i, sample_uv in enumerate(sample_uvs):
    #     img = images_train[0]
    #     gen_pixel = img[sample_uv[1], sample_uv[0]]
    #     gen_pixel = images_train[0, sample_uv[1], sample_uv[0]]
    #     pixel = pixels[i]
    #     assert np.isclose(gen_pixel, pixel).all()
    # assert np.all(images_train[0, sample_uvs[:,1], sample_uvs[:,0]] == pixels[uvs_start:uvs_end])
    data = {"rays_o": rays_o, "rays_d": rays_d}
    # points = sample_along_rays(data["rays_o"], data["rays_d"], random=True)
    points = sample_along_rays(ray_color_tensors, near=2.0, far=6.0, samples=32, perturb=True)
    H, W = images_train.shape[1:3]
    K = intrinsic_K(focal, H, W)
    # ---------------------------------------

    server = viser.ViserServer(share=True)
    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image
        )
    for i, (o, d) in enumerate(zip(data["rays_o"], data["rays_d"])):
        positions = np.stack((o, o + d * 6.0))
        server.add_spline_catmull_rom(
            f"/rays/{i}", positions=positions,
        )
    server.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.03,
    )
    time.sleep(1000)  
def calibrate_volume():
    torch.manual_seed(42)
    sigmas = torch.rand((10, 64, 1))
    rgbs = torch.rand((10, 64, 3))
    step_size = (6.0 - 2.0) / 64
    rendered_colors = volume_rendering(sigmas, rgbs, step_size)

    correct = torch.tensor([
        [0.5006, 0.3728, 0.4728],
        [0.4322, 0.3559, 0.4134],
        [0.4027, 0.4394, 0.4610],
        [0.4514, 0.3829, 0.4196],
        [0.4002, 0.4599, 0.4103],
        [0.4471, 0.4044, 0.4069],
        [0.4285, 0.4072, 0.3777],
        [0.4152, 0.4190, 0.4361],
        [0.4051, 0.3651, 0.3969],
        [0.3253, 0.3587, 0.4215]
    ])
    assert torch.allclose(rendered_colors, correct, rtol=1e-4, atol=1e-4)
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
    EPOCH = 10
    LAYERS = 8
    LEARNING_RATE = 5e-4
    mp.set_start_method('spawn')
    nerf = DeepNeRF(learning_rate=LEARNING_RATE)
    # dataset = NerfSingularDataSet(data=images_train, num_samples=1000, num_workers = 4, f = focal, c2w = c2ws_train, im_height=200, im_width=200)
    dataset = NerfDataSet(data=images_train, num_samples=10000, num_workers = multiprocessing.cpu_count(), f = focal, c2w = c2ws_train, im_height=200, im_width=200)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    psnr = 0.0
    for i in tqdm(range(EPOCH), "Training iteration:"):
        for idx, batch in enumerate(tqdm(dataloader, "Going through batch")):
            # rays_o = batch[:, 1:, 0].numpy()
            # rays_d = batch[:, 1:, 1].numpy()
            batch = batch[0]
            actual_colors = batch[:, 1:, 2].float()
            points = sample_along_rays(batch, near=2.0, far=6.0, samples=64, perturb=True, with_rays=True)
            coords = torch.from_numpy(points[:, :3]).float()
            ray_ds = torch.from_numpy(points[:, 3:]).float()
            nerf.train(coords, ray_ds, actual_colors)
            psnr += nerf.get_psnrs()[-1]
            if idx % 20 == 0:
               print(f"psnr current {idx}: {psnr}") 
        print(f"psnr final {i}: {psnr}")
        psnr = 0.0

            
    # calibrate_part1()
    # calibrate_part2()
    # calibrate_volume()
    nerf.save_model(osp.join(osp.abspath(osp.dirname(__file__)), "checkpoints", f"deep_nerf_epoch{EPOCH}_LR{LEARNING_RATE}_LAYER{LAYERS}.pth"))
    #metrics
    train_psnrs = nerf.get_psnrs()
    nerf.psnrs = []
    # pred = test(img1, nerf)
    psnrs = nerf.get_psnrs()
    # plt.imshow(img1)
    # plt.show()
    
    plt.suptitle(f"PNSRS and Image test w/ num_layers: {LAYERS}, " \
            + f"learning rate: {LEARNING_RATE}, epochs: {EPOCH}")
    plt.plot(range(len(train_psnrs)), train_psnrs)
    plt.title("PNSRS over train iterations")
    img_folder = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), "images")
    psnr_plot = osp.join(img_folder, "deep.png")
    plt.savefig(psnr_plot)
    
   
    
    
    
    
    
def main():
    if PART_1:
        part_1()
    if PART_2:
        part_2()
    
    
if __name__ == "__main__":
    main()


