import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
from skimage.draw import polygon
from concurrent.futures.thread import ThreadPoolExecutor
import matplotlib.pyplot as plt
from scipy.ndimage import shift
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
def get_pts(im, num_pts=4):
    plt.imshow(im)
    plt.title("Click at least 4 points")
    pts = plt.ginput(num_pts)
    plt.close()
    return pts
def ncc(vec1, vec2):
    """Compute Normalized Cross-Correlation between two vectors

    Args:
        vec1 (_type_): vector of image 1
        vec2 (_type_): vector of image 2
        vec1.shape = vec2.shape

    Returns:
        int: value of dot product of normalized vectors
    """
    mean1 = np.mean(vec1)
    mean2 = np.mean(vec2)
    cen_vec1 = vec1 - mean1
    cen_vec2 = vec2 - mean2
    f_norm1 = np.linalg.norm(cen_vec1)
    f_norm2 = np.linalg.norm(cen_vec2)
    return np.dot(cen_vec1, cen_vec2) / (f_norm1 * f_norm2)
def simple_align(im_1, im_2, displacement: int, start_trans: list[int] = [0, 0]) -> list[int]:
    """Aligns images via Normalized Cross-Correlation

    Args:
        im_1 (np.array): 2D numpy array w/ first image as base
        im_2 (np.array): 2D numpy array w/ second image comparing to base
        displacement (int): number of pixel displacements to search up until 
    
    Returns:
        list[int]: best translation to move im_2 to im_1
    """
    assert im_1.shape[0] == im_2.shape[0]
    assert im_1.shape[1] == im_2.shape[1]
    # base_avg = np.mean(im_2)
    # comp_avg = np.mean(im_1)
    # vec1 = im_1.flatten()
    vec2 = im_2.flatten()
    ddist: list[int] = list(range(-displacement, displacement + 1))
    max_val = -(float('inf'))
    best_trans = start_trans[:]
    for diff_x in tqdm(ddist, "aligning w/ max displacement: {}".format(displacement)):
        for diff_y in ddist:
            vec1 = shift2(im_1, (start_trans[0] + diff_x, start_trans[1] + diff_y)).flatten()
            
            # plt.imshow(img1)
            # plt.show()
            ncc_val = ncc(vec1, vec2)
            if ncc_val > max_val:
                best_trans[0] = start_trans[0] + diff_x
                best_trans[1] = start_trans[1] + diff_y
                max_val = ncc_val
    return best_trans
def shift2(matrix, shift_values):
    """given matrix and shift values, shift matrix and return shifted matrix

    Args:
        matrix (2D np.array): 2D img matrix
        shift_values (list or tuple): shift deltas in form (dx, dy)

    Returns:
        _type_: _description_
    """
    return shift(matrix, shift_values, mode='constant', cval=0) 
def refine_by_ncc(im1, im2, pt1, pt2, size=6):
    np.pad(im1, pad_width=size, mode='constant', constant_values=0)
    max_val = -(float('inf'))
    best_pt1 = pt1
    for dx in range(-size, size + 1):
        for dy in range(-size, size + 1):
            new_pt1_x = pt1[0] + dx
            new_pt1_y = pt1[1] + dy
            vec1 = im1[new_pt1_y - size: new_pt1_y + size + 1, new_pt1_x - size: new_pt1_x + size + 1, :].flatten()
            vec2 = im2[pt2[1] - size: pt2[1] + size + 1, pt2[0] - size: pt2[0] + size + 1, :].flatten()
            ncc_val = ncc(vec1, vec2)
            if ncc_val > max_val:
                max_val = ncc_val
                best_pt1 = [new_pt1_x, new_pt1_y]
    return best_pt1, pt2
    
def warpImage(im, H):
    h, w = im.shape[:2]
    box_corners = [[0, 0], 
                   [0, h],
                   [w, 0],
                   [w, h]]
    box_arr = np.array(box_corners)
    box_arr_pts = np.hstack([np.ones(4)])
    new_bounds = np.round(H @ box_arr_pts)
    lst = []
    for i in range(4):
        p = new_bounds[:, i]
        lst.append([p[0] / p[2], p[1] / p[2]])
    new_bounds = np.array(lst).T
    xMax = np.max(new_bounds[0, :])
    xMax = max(xMax, im.shape[1])
    yMax = np.max(new_bounds[1, :])
    yMax = max(yMax, im.shape[0])
    xMin = np.min(new_bounds[0, :])
    yMin = np.min(new_bounds[1, :])
    shape_max = (yMax + (yMin if yMin < 0 else 0), xMax + (xMin if xMin < 0 else 0))
    shape_min = (yMin if yMin > 0 else 0, xMin if xMin > 0 else 0)
    
    r, c = polgyon(np.squeeze(new_bounds[1, :]), np.squeeze(new_bounds[0, :]))
def main():
    im1 = ...
    im2 = ...
    points = int(input("How many points?"))
    pts1 = get_pts(im1, points)
    pts2 = get_pts(im2, points)
    
    

    
                   
    
    
    
    
    
def predict_pt(H, pt1):
    pt2 = H @ pt1
    return (pt2 / pt2[2])