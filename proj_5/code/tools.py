import torch
import numpy as np
def transform_c2w(c2w: torch.Tensor, x_c: torch.Tensor) -> torch.Tensor:
    trans = torch.matmul(c2w, x_c)
    assert(torch.equal(x_c, torch.matmul(torch.inverse(c2w), torch.matmul(c2w, x_c))))
    return trans
def transform_c2w_np(c2w: np.array, x_c: np.array) -> torch.Tensor:
    trans = c2w @ x_c
    assert(torch.equal(x_c, torch.matmul(torch.inverse(c2w), torch.matmul(c2w, x_c))))
    return trans
def pixel_to_camera(K: torch.Tensor, uv: torch.Tensor, s: float) -> torch.Tensor:
    x_c = torch.matmul(torch.inverse(K), s * uv)
    return x_c
def pixel_to_ray(K: torch.Tensor, c2w: torch.Tensor, uv: torch.Tensor) -> tuple[torch.Tensor]:
    R = K[:3, :3]
    t = K[:3, 3]
    r0: torch.Tensor = -1 * torch.matmul(torch.inverse(R), t)
    x_c = pixel_to_camera(K, uv, 1)
    x_w = transform_c2w(c2w, x_c)
    rd: torch.Tensor = (x_w - r0) / torch.norm(x_w - r0)
    return r0, rd
def intrinsic_K(f: float, height: int, width: int):
    sigma_x = width / 2
    sigma_y = height / 2
    k = torch.tensor([[f, 0, sigma_x],
                      [0, f, sigma_y],
                      [0, 0, 1]])
    return k