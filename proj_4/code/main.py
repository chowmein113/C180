import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
from skimage.draw import polygon
from concurrent.futures.thread import ThreadPoolExecutor
import matplotlib.pyplot as plt
from scipy.ndimage import shift, map_coordinates
import skimage.io as skio
import os
import os.path as osp
import cv2
#Global
MAX_WORKERS = 7
RECTIFY = True
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
    # ata = A.T @ A
    # atb = A.T @ b
    # h = np.linalg.inv(ata) @ atb
    h, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)
    h = np.append(h, 1)
    return h.reshape(3, 3)

def apply_alpha(im: np.array) -> np.array:
    #TODO: FIX
    alpha_im = im
    if len(im.shape) == 3:
        if len(im.shape[2]) == 3:
            alpha_im = np.dstack([im, np.ones_like(im[:, :, 0]) * 255])
            
    return alpha_im
def get_pts(im, num_pts=4):
    plt.imshow(im)
    plt.title(f"Click {num_pts} points")
    pts = plt.ginput(num_pts, timeout=0)
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
                   [0, h - 1],
                   [w - 1, h - 1],
                   [w - 1, 0]]
    box_arr = np.array(box_corners)
    box_arr_pts = (np.hstack([box_arr, np.ones((4, 1))]))
    print(f"box_arr_pts: {box_arr_pts}")
    new_bounds = (H @ box_arr_pts.T)
    print(f"new bounds: pre{new_bounds}")
    lst = []
    for i in range(4):
        p = new_bounds[:, i]
        lst.append([int(np.round(p[0] / p[2])), int(np.round(p[1] / p[2]))])
    new_bounds = np.array(lst).T
    print(f"new bounds: {new_bounds}")
    xMax = np.max(new_bounds[0, :])
    # xMax = max(xMax, im.shape[1])
    yMax = np.max(new_bounds[1, :])
    # yMax = max(yMax, im.shape[0])
    xMin = np.min(new_bounds[0, :])
    yMin = np.min(new_bounds[1, :])
    off_x = -1 * min(xMin, 0)
    off_y = -1 * min(yMin, 0)
    shape_max = (int(max(yMax + off_y, im.shape[0])), int(max(xMax + off_x, im.shape[1])))
    shape_min = (0, 0)
    ydist = yMin if yMin < 0 else 0
    xdist = xMax if xMin < 0 else 0
    new_img = np.zeros(list(shape_max) + [3])
    
    
    o_r, o_c = polygon(box_arr[:, 1], box_arr[:, 0])
    r, c = polygon(np.squeeze(new_bounds[1, :]), np.squeeze(new_bounds[0, :]))
    # r = np.clip(np.array(r), yMin, yMax - 1)
    # c = np.clip(np.array(c), xMin, xMax - 1)
    # r += off_y
    # c += off_x
    pts = np.row_stack((c, r, np.ones(r.shape[0])))
    H_inv = np.linalg.inv(H)
    pts_in_im = (H_inv @ pts)
    im_x = np.round(pts_in_im[0, :] / pts_in_im[2, :]).astype(int)
    im_y = np.round(pts_in_im[1, :] / pts_in_im[2, :]).astype(int)
    pts[0, :] += off_x
    pts[1, :] += off_y
    pts = pts.astype(int)
    
    
    ###not here
    ins = np.where((im_y >= 0) & (im_y < im.shape[0]) & (im_x >= 0) & (im_x < im.shape[1]))
    ###
    f, axs = plt.subplots(1,2)
    from_im = im[im_y[ins], im_x[ins]]
    img_copy = np.zeros(im.shape[:2])
    img_copy[o_r, o_c] = 3
    img_copy[im_y[ins], im_x[ins]] = 1
    new_im_cop = np.zeros(new_img.shape[:2])
    new_im_cop[pts[1, :], pts[0, :]] = 1
    new_im_cop[pts[1, :][ins], pts[0, :][ins]] = 2
    # img_copy = (img_copy * 255).astype(np.uint8)
    ma = np.max(img_copy)
    mi = np.min(img_copy)
    axs[0].imshow(img_copy)
    axs[1].imshow(new_im_cop)
    plt.show()
    orr = pts[1, :]
    print(f"or max: {np.max(orr)} and or min: {np.min(orr)}")
    occ = pts[0, :]
    print(f"oc max: {np.max(occ)} and oc min: {np.min(occ)}")
    new_img[orr[ins], occ[ins]] = from_im
    ma = np.max(new_img)
    mi = np.min(new_img)
    
    
    return (new_img * 255).astype(np.uint8)
    # return (canvas * 255).astype(np.uint8)

def predict_pt(H, pt1):
    pt2 = H @ pt1
    return (pt2 / pt2[2])   
def get_avg_dist(pts):
    sum_dist = 0
    for i in range(pts.shape[0]):
        for j in range(i - 1, i + 2):
            pt1 = pts[i, :]
            pt2 = pts[j % pts.shape[0], :]
            sum_dist += np.linalg.norm(pt2 - pt1)
    return sum_dist / pts.shape[0]


def main():
    img_folder = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "images")
    im1 = skio.imread(osp.join(img_folder, "walk.png")) / 255
    
    # points = int(input("How many points? "))
    # pts1 = np.round(np.array(get_pts(im1, points)))
    if RECTIFY:
        points = 4
        pts1 = np.round(np.array(get_pts(im1, points))).astype(int)
        # off_x = np.min(pts1[:, 0])
        # off_y = np.min(pts1[:, 1])
        off_x = 100
        off_y = 100
        # s = get_avg_dist(pts1)
        # s = int(np.linalg.norm(pts1[1, :] - pts1[0, :]))
        # s = im1.shape[0]
        s = 1000
        # pts2 = np.array([[0, 0],
        #                 [0.5, 0],
        #                 [1, 0],
        #                 [1, 0.5],
        #                 [1, 1],
        #                 [0.5, 1],
        #                 [0, 1],
        #                 [0, 0.5]]) * s
        pts2 = np.array([[0., 0.], #ul
                        [1., 0.], #ur
                        [0., 1.],#bl
                        [1., 1.]]) * s  #br
        # pts2 = np.array([[1., 0.], #ur
        #                 [1., 1.], #br
        #                 [0., 1.],#bl
        #                 [0., 0.]]) * s #ul
        # pts2[:, 0] += off_x
        # pts2[:, 1] += off_y
        H = computeH(pts1, pts2)
        # H2 = cv2.findHomography(pts1, pts2)
        print(f"H: {H}")
        # print(f"H2: {H2}")
        new_img = warpImage(im1, H)
        # new_img = warp_image_rectify(im1, H)
        # new_img2 = warpImage2(im1, im1, H)
        f, axs = plt.subplots(1,2)
        axs[0].set_title("Original Image")
        axs[0].imshow(im1)
        axs[1].set_title("Frontal Parallel Rectified Image")
        axs[1].imshow(new_img)
        # axs[2].imshow(new_img2)
        # plt.tight_layout()
        plt.show()
        
    # im2 = ...
    # pts2 = get_pts(im2, points)
if __name__ == "__main__":
    main()
