import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor

#Global
MAX_WORKERS = 7
def computeH(pts1, pts2) -> np.array:
    """Return homography matrix that maps pts1 to pts 2
    p2 = Hp1
    H = p2 inv(p1) or inv(p1T) p2T = HT

    Args:
        pts1 (_type_): _description_
        pts2 (_type_): _description_

    Returns:
        np.array: _description_
    """
    A = []
    b = []
    #h = [a, b, c, d, e, f, g, h, i]
    for p1, p2 in zip(pts1, pts2):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        b.extend([p2[0], p2[1]])
        A.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        A.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
    A = np.array(A)
    b = np.array(b)
    ata = A.T @ A
    atb = A.T @ b
    h = np.linalg.inv(ata) @ atb
    return h.reshape(3, 3)
def apply_alpha(im: np.array) -> np.array:
    alpha_im = im
    if len(im.shape) == 3:
        if len(im.shape[2]) == 3:
            alpha_im = np.dstack([im, np.ones_like(im[:, :, 0]) * 255])
            
    return alpha_im
def warpImage(im, H):
    h, w = im.shape[:2]
    box_corners = [[0, 0], 
                   [0, h],
                   [w, 0],
                   [w, h]]
    box_arr = np.array(box_corners)
    bo_arr_pts = np.hstack([np.ones(4)])
def predict_pt(H, pt1):
    pt2 = H @ pt1
    return (pt2 / pt2[2])