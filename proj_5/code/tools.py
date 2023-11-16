import torch
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
import multiprocessing
def transform_c2w(c2w: torch.Tensor, x_c: torch.Tensor) -> torch.Tensor:
    trans = torch.matmul(c2w, x_c)
    # remade = torch.matmul(torch.inverse(c2w), trans) #should be like x_c
    # assert(torch.equal(x_c, torch.matmul(torch.inverse(c2w), trans)))
    # assert torch.allclose()
    return trans
def transform_c2w_np(c2w: np.array, x_c: np.array) -> torch.Tensor:
    trans = c2w @ x_c
    assert(np.array_equal(x_c, np.linalg.inv(c2w) @ (trans)))
    return trans
def pixel_to_camera(K: torch.Tensor, uv: torch.Tensor, s: float) -> torch.Tensor:
    uv_homo = torch.hstack((uv, torch.ones((1,))))
    x_c = torch.matmul(torch.inverse(K), s * uv_homo)
    return x_c
def pixel_to_camera_np(K: np.array, uv: np.array, s: float) -> torch.Tensor:
    x_c = np.linalg.inv(K) @ (s * uv)
    return x_c
def pixel_to_ray(K: torch.Tensor, c2w: torch.Tensor, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    # r0: torch.Tensor = -1 * torch.matmul(torch.inverse(R), t)
    r0: torch.Tensor = c2w[:3, 3]
    x_c = pixel_to_camera(K, uv, 1)
    x_c_homo = torch.hstack([x_c, torch.ones(1)])
    x_w_homo = transform_c2w(c2w, x_c_homo)
    x_w = x_w_homo[:3]
    rd: torch.Tensor = (x_w - r0) / torch.norm(x_w - r0)
    return r0, rd
def intrinsic_K(f: float, height: int, width: int):
    sigma_x = width / 2
    sigma_y = height / 2
    k = torch.from_numpy(np.array([[f, 0, sigma_x],
                      [0, f, sigma_y],
                      [0, 0, 1]]))
    return k
def point_from_ray(r0: torch.Tensor, rd: torch.Tensor, t) -> torch.Tensor:
    return r0 + rd * t
def sample_along_ray(r0, rd, near: float, far: float, samples: int = 32, perturb=False, with_rays = False):
    t_set = np.linspace(near, far, samples)
    t_width = 0.1
    if perturb:
        t_set += np.random.rand(*t_set.shape) * t_width
    if with_rays:
        points = [torch.hstack((point_from_ray(r0, rd, t), rd)).numpy() for t in t_set]
    else:
        points = [point_from_ray(r0, rd, t).numpy() for t in t_set]
    return points
def separate_ray_pairs(ray_pairs: torch.Tensor):
    #a ray pair format in torch tensor list of ray pairs
    #[[cam_num, cam_num, cam_num]
    #[r0,         rd,      rgb]
    #[r0,         rd,      rgb]
    #[r0,         rd,      rgb]]
    lst = []
    for i in range(ray_pairs.size()[0]):
        ray_pair = ray_pairs[i]
        r0 = ray_pair[1:, 0]
        rd = ray_pair[1:, 1]
        pixel = ray_pair[1:, 2]
        cam_num = ray_pair[0, 0].item()
        lst.append((cam_num, r0, rd, pixel))
def worker(ray_pairs, near: float, far: float, samples: int = 32, perturb=False, with_rays=False):
    # ray_pairs = ray_pair_chunks[i]
    points = []
    # for i in tqdm(range(ray_pairs.size(0)), "Generating points: "):
    #     ray_pair = ray_pairs[i]
    #     r0 = ray_pair[1:, 0]
    #     rd = ray_pair[1:, 1]
    #     points.extend(sample_along_ray(r0, rd, near, far, samples, perturb, with_rays))
    # return np.array(points)
    t_set = np.linspace(near, far, samples)
    t_width = 0.1
    if perturb:
        t_set += np.random.rand(*t_set.shape) * t_width
    t_set = torch.from_numpy(t_set)
    # ones = torch.ones_like(t_set)
    r0s = ray_pairs[:, 1:, 0]
    rds = ray_pairs[:, 1:, 1]
    rd_expanded = rds.repeat(samples, 1)
    r0_expanded = r0s.repeat(samples, 1)
    reshaped_t_set = t_set.repeat_interleave(ray_pairs.size(0))

    # Calculate points
    points = r0_expanded + rd_expanded * reshaped_t_set.unsqueeze(-1)  # Shape will be [num_rays, num sampl, 3]

    if with_rays:
        
        # rds_expanded = rds.unsqueeze(1).expand(-1, samples, -1)
        points = torch.cat([points, rd_expanded], dim=-1) 

    return points.numpy() 
 

def sample_along_rays(ray_pairs: torch.Tensor, near: float, far: float, samples: int = 32, perturb=False, with_rays=False):
    #a single ray pair format in torch tensor list of ray pairs
    #[[cam_num, cam_num, cam_num]
    #[r0,         rd,      rgb]
    #[r0,         rd,      rgb]
    #[r0,         rd,      rgb]]
    # points = []
    # processes = []
    # result_queue = mp.Queue()
    results = []
    num_processes = 2
    assert ray_pairs.size(0) % num_processes == 0
    torch.set_num_threads(1)
    ray_pair_chunks = ray_pairs.view(num_processes, ray_pairs.size(0) // num_processes, 4, 3)
    ray_pair_chunks.share_memory_()
    with multiprocessing.Pool(num_processes) as pool:
        all_args = [(ray_pair_chunks[i], near, far, samples, perturb, with_rays) for i in range(num_processes)]
        results = pool.starmap(worker, all_args)
    # results = worker(ray_pairs, near, far, samples, perturb, with_rays)
    return np.vstack(results)
    # return np.array(results)
    # for i in tqdm(range(num_processes), "making sub process: "):
    #     p = mp.Process(target=worker, args=(i, ray_pair_chunks, near, far, result_queue, samples, perturb, with_rays))
    #     p.start()
    #     processes.append(p)
        
    # for p in tqdm(processes, "joining sub processes..."):
    #     p.join()
    # results = []
    # while not result_queue.empty():
    #     item = result_queue.get()
    #     if isinstance(item, Exception):
    #         raise item  # Re-raise exception from the subprocess
    #     results.append(item)
    # return np.vstack(results)
    # for i in tqdm(range(ray_pairs.size(0)), "Generating points: "):
    #     ray_pair = ray_pairs[i]
    #     r0 = ray_pair[1:, 0]
    #     rd = ray_pair[1:, 1]
    #     points.extend(sample_along_ray(r0, rd, near, far, samples, perturb, with_rays))
    # return np.array(points)

    
def volume_rendering(sigmas, rgbs, step_size):
    # Compute the prob from the density/sigma values
    prob_i = 1.0 - torch.exp(-sigmas * step_size)
    T_i_sum = torch.exp(-1 * torch.cumsum(sigmas[:, :-1], dim=1) * step_size)
    last = torch.ones_like(sigmas[:, :1])
    # T_i = torch.cat((T_i, last), dim=1)
    T_i = torch.cat((last, T_i_sum), dim=1)
    vol = prob_i * T_i
    rendered_colors = torch.sum(vol * rgbs, dim=1)
    return rendered_colors
