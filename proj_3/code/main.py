import array
from pickle import FALSE
from scipy.spatial import Delaunay
import numpy as np
import skimage
import skimage.io as skio
from skimage.draw import polygon
from tools import get_correspondence_pts_from_JSON, load_pts, load_dataset, load_asf_dataset
import os.path as osp
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import cv2
from concurrent.futures.thread import ThreadPoolExecutor
import re


#GLOBAL VARIABLES
MAX_WORKERS = 15
MIDFACE = True
MORPH = False
MEANFACE = True
CARICATURE = True

def compute_mean_pts(pts_list1: list[list[int]], pts_list2: list[list[int]]) -> list[float]:
    ans = []
    assert len(pts_list1) == len(pts_list2)
    for i in range(len(pts_list1)):
        
        assert len(pts_list1[i]) == 2 and len(pts_list2[i])
        img1_pt = pts_list1[i]
        img2_pt = pts_list2[i]
        ans.append([(img1_pt[0] + img2_pt[0]) / 2, (img1_pt[1] + img2_pt[1]) / 2])
    return ans

def compute_affine(tri_pts1: np.array, tri_pts2: np.array) -> np.array:
    """Given two sets of triangles, compute affine transformation matrix values
    in homogenous coords

    Args:
        tri_pts1 (list[list[int]]): list of points for triangle 1 src
        tri_pts2 (list[list[int]]): list of points for triangle 2 destination

    Returns:
        np.array: Transformation matrix in homogenous 3D coords for tri1 to tri2
    """
    
    ones = np.ones((tri_pts1.shape[0], 1))
    X1 = np.hstack([tri_pts1, ones]).T
    if np.linalg.det(X1) == 0 or True:
        inv_X1 = np.linalg.inv(X1)
    else:
        inv_X1 = np.linalg.pinv(X1)
    T = np.hstack([tri_pts2, np.ones((tri_pts2.shape[0], 1))]).T @ inv_X1
    # ones = np.ones((tri_pts1.shape[0], 1))
    # X1 = np.hstack([tri_pts1, ones])
    # T = np.linalg.solve(np.hstack([tri_pts2, np.ones((tri_pts2.shape[0], 1))]), X1)
    #Tx_1 = x_2
    # (x_1^T(T^T)) = x_2^T
    return T
def compute_inverse_affine(tri_pts1: np.array, tri_pts2: np.array) -> np.array:
    ones = np.ones((tri_pts1.shape[0], 1))
    return np.linalg.solve(np.hstack([tri_pts2, np.ones((tri_pts2.shape[0], 1))]),
                           np.hstack([tri_pts1, ones])).T

def compute_mid_face(im1: np.array, im2: np.array, im1_pts: np.array, 
          im2_pts: np.array, tri: Delaunay):
    return morph(im1, im2, im1_pts, im2_pts, tri, 0.5, 0.5)
def linear_interp2d(im, target_pts):
    """Calculate 

    Args:
        im (_type_): _description_
        target_pts (_type_): _description_
    """
    pairs = np.column_stack(target_pts)
    height = im.shape[0]
    width = im.shape[1]
    #TODO: implement if nearest not good enough, computationally expensive if bilinear
    return
    
def compute_tri_move(tri_indices, vec1, vec2, warp_pts, base1, base2, im1, im2):
        T_img1 = compute_affine(vec1[tri_indices], warp_pts[tri_indices])
        T_img2 = compute_affine(vec2[tri_indices], warp_pts[tri_indices])
        
        c, r = polygon(warp_pts[tri_indices][:, 0], warp_pts[tri_indices][:, 1])
        coord_pts = (c, r)
        
        #extend coords to 3d for homogenous matrix use and put coord back in x, y format from r, c (i.e. y, x)
        coord_vec_2d = [i.reshape(i.shape[0], 1) for i in coord_pts]
        col1 = np.ones((coord_vec_2d[0].shape[0], 1)).astype(np.int64)
        coord_vec_3d = np.hstack(list(coord_vec_2d) + [col1])
        
        new_pts1 = tuple(((np.linalg.inv(T_img1) @ coord_vec_3d.T)[:2, :]).astype(int))[::-1]
        new_pts2 = tuple(((np.linalg.inv(T_img2) @ coord_vec_3d.T)[:2, :]).astype(int))[::-1]
        
        
        coord_pts = coord_pts[::-1]
        
        
        
        to_assign = map_coordinates(im1, list(new_pts1), order=0)
        base1[coord_pts] = to_assign
        to_assign2 = map_coordinates(im2, new_pts2, order=0)
        base2[coord_pts] = to_assign2
        return base1, base2    
def morph(im1: np.array, im2: np.array, im1_pts: np.array, 
          im2_pts: np.array, tri: Delaunay, warp_frac: float, 
          dissolve_frac: float) -> np.array:
    max_height = max(im1.shape[0], im2.shape[0])
    max_width = max(im1.shape[1], im2.shape[1])
    max_shape = (max_height, max_width, im1.shape[2])
    
    vec1 = im1_pts
    vec2 = im2_pts
    warp_pts = warp_frac * vec1 + (1 - warp_frac) * vec2
    # tri = Delaunay(warp_pts)
    # warp_pts[0] /= np.max(warp_pts[0])
    # warp_pts[1] /= np.max(warp_pts[1])
    
    # base1 = np.zeros(max_shape)
    # base1[:im1.shape[0], :im1.shape[1]] = im1
    # base2 = np.zeros(max_shape)
    # base2[:im2.shape[0], :im2.shape[1]]
    
    base1 = im1.copy()
    base2 = im2.copy()
    
    
    for tri_indices in tri.simplices:
        T_img1 = compute_affine(vec1[tri_indices], warp_pts[tri_indices])
        T_img2 = compute_affine(vec2[tri_indices], warp_pts[tri_indices])
        
        c, r = polygon(warp_pts[tri_indices][:, 0], warp_pts[tri_indices][:, 1], im1.shape[:2])
        coord_pts = (c, r)
        
        #extend coords to 3d for homogenous matrix use and put coord back in x, y format from r, c (i.e. y, x)
        coord_vec_2d = [i.reshape(i.shape[0], 1) for i in coord_pts]
        col1 = np.ones((coord_vec_2d[0].shape[0], 1)).astype(np.int64)
        coord_vec_3d = np.hstack(list(coord_vec_2d) + [col1])
        #use nearest neighbor by rounding with numpy
        to_be_mul = coord_vec_3d.T
        mat = np.linalg.inv(T_img1)
        res0 = (mat @ to_be_mul)
        dt = np.linalg.det(T_img1)
        res = res0[:2, :]
        round_res = np.ndarray.round(res).astype(np.int32)
        new_pts1 = tuple(round_res)[::-1]
        new_pts1 = tuple(np.ndarray.round((mat @ to_be_mul)[:2, :]).astype(int))[::-1]
        new_pts2 = tuple(np.ndarray.round((np.linalg.inv(T_img2) @ coord_vec_3d.T)[:2, :]).astype(int))[::-1]
        
        
        coord_pts = coord_pts[::-1]
        
        base1[coord_pts] = im1[new_pts1]
        base2[coord_pts] = im2[new_pts2]
        
        
    hybrid_img = dissolve_frac * base1 + (1 - dissolve_frac) * base2
    
    return (hybrid_img * 255).astype(np.uint8)
        

def compute_triangles(pts_list1: list[list[int]], pts_list2: list[list[int]]) ->Delaunay:
      mean_pts = compute_mean_pts(pts_list1, pts_list2)
      return Delaunay(mean_pts)
def compute_tri_with_array(pts1: np.array, pts2: np.array):
    return Delaunay((pts1 + pts2) / 2)
def compute_morph(img1: np.array, img2: np.array, 
                  pts_list1: list[list[int]], 
                  pts_list2: list[list[int]]):
    
    return
def make_morph_video(im1: np.array, im2: np.array, im1_pts: np.array, 
          im2_pts: np.array, tri: Delaunay, duration: float, fps: float) -> list[np.array]:
    """Generate list of different frames to show complete morph transition from IM1 to IM2.

    Args:
        im1 (np.array): numpy image array
        im2 (np.array): numpy image array
        im1_pts (np.array): correspondence points for IM1
        im2_pts (np.array): correspondence points for IM2
        tri (Delaunay): Delaunay object of traingles calculated from Correspondence points
        duration (float): video duration in seconds
        fps (float): frame per second of video transition

    Returns:
        list[np.array] : list of images to be made into video
    """
    total_frames: int = int((fps * duration) // 1)
    frames = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        threads = []
        for frame in tqdm(range(0, total_frames), "Making morph frames"):
            warp_frac: float = frame / (total_frames - 1)
            dissolve_frac: float = warp_frac
            threads.append(executor.submit(morph, im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac))
        results = [thread.result() for thread in tqdm(threads, "Getting Morph Results")]
        for i, result in tqdm(enumerate(results), "Going through frame results"):
            frames.append(result)
    # for frame in tqdm(range(0, total_frames), "Making morph frames"):
    #         warp_frac: float = frame / (total_frames - 1)
    #         dissolve_frac: float = warp_frac
    #         frames.append(morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac))
    return frames
def create_cv2_video(frames: list[np.array], out_path: str, fps: float = 30., debug=False):
    """Given a list of FRAMES, make video and save to OUT_PATH

    Args:
        frames (list[np.array]): list of numpy image arrays
        out_path (str): file path str to save video to
        fps (float, optional): desired frame per sec. Defaults to 30..
    """
    assert len(frames) > 0
    height, width = frames[0].shape[:2]
    fork = cv2.VideoWriter_fourcc(*'mp4v')
    vid = cv2.VideoWriter(out_path, fork, fps, (width, height))
    for frame in tqdm(frames, "Writing frames to video"):
        if debug:
            plt.imshow(frame)
            plt.title("frame")
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vid.write(frame_bgr)
    vid.release()

def compute_avg_key_pts(dataset, get_img=False):
    # avg_pts = np.zeros(1)
    # avg_img = np.zeros(1)
    avg_pts = []
    avg_img = []
    iters = 0
    for fn, data in dataset.items():
        avg_pts.append(np.array(data.get('pts')))
        avg_img.append(np.array(data.get('img')))
        # if iters == 0:
        #     iters = 1
        #     avg_pts = np.array(data.get('pts'))
        #     if get_img:
        #         avg_img = np.array(data.get('img'))
        # else:
        #     iters += 1
        #     avg_pts += np.array(data.get('pts'))
        #     if get_img:
        #         avg_img += np.array(data.get('img'))
    
            
    
    return np.mean(avg_pts, axis=0), np.mean(avg_img, axis=0)
    # return avg_pts / iters, avg_img / iters
def morph_shape(im1, im1_pts: np.array, 
          im1_new_pts: np.array, tri: Delaunay):
    vec1 = im1_new_pts
    warp_pts = im1_pts
    base1 = np.zeros(im1.shape)
    for tri_indices in tri.simplices:
        #avg img to og image
        T_img1 = compute_affine(vec1[tri_indices], warp_pts[tri_indices])
        
        c, r = polygon(warp_pts[tri_indices][:, 0], warp_pts[tri_indices][:, 1])
        coord_pts = (c, r)
        
        #extend coords to 3d for homogenous matrix use and put coord back in x, y format from r, c (i.e. y, x)
        coord_vec_2d = [i.reshape(i.shape[0], 1) for i in coord_pts]
        col1 = np.ones((coord_vec_2d[0].shape[0], 1)).astype(np.int64)
        coord_vec_3d = np.hstack(list(coord_vec_2d) + [col1])
        #use nearest neighbor by rounding with numpy
        to_be_mul = coord_vec_3d.T
        # mat = np.linalg.inv(T_img1)
        mat = compute_inverse_affine(vec1[tri_indices], warp_pts[tri_indices])
        res = (mat @ to_be_mul)
        res = res[:2, :]
        round_res = np.ndarray.round(res).astype(np.int32)
        new_pts1 = tuple(round_res)[::-1]
        new_pts1 = tuple(np.ndarray.round((mat @ to_be_mul)[:2, :]).astype(int))[::-1]
        
        coord_pts = coord_pts[::-1]
        # f, axs = plt.subplots(1, 2)
        # name1 = ("im1_name")
        # name2 = ("im2_name")
        # vec1 = vec1[tri_indices]
        # axs[0].set_title(f"{name1} Delaunay triangulation with avg points")
        # axs[0].imshow(im1)
        # axs[0].scatter(vec1[:, 0], vec1[:, 1])
        # # axs[0].triplot(vec1[:, 0], vec1[:, 1], tri.simplices)
        # axs[1].set_title(f"{name2} Delaunay triangulation with avg points")
        # axs[1].imshow(im1)
        # vec2 = warp_pts[tri_indices]
        # axs[1].scatter(vec2[:, 0], vec2[:, 1])
        # axs[1].triplot(vec2[:, 0], vec2[:, 1], tri_indices)
        plt.show()
        base1[coord_pts] = im1[new_pts1]
        
    return base1
def turn_dataset_into_avg(dataset, avg_pts, avg_img, tri=None, warp_frac = 0.0, dissolve_frac = 0.8):
    new_avg = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        threads = {}
        for fn, data in dataset.items():
            pts1 = data.get('pts')
            im1 = data.get('img')
            if pts1 is not None and im1 is not None:
                # tri = compute_tri_with_array(pts1, avg_pts)
                if tri is None:
                    tri = Delaunay(avg_pts)
                ###Test
                # f, axs = plt.subplots(1, 2)
                # name1 = ("im1_name")
                # name2 = ("im2_name")
                # vec1 = np.array(pts1)
                # axs[0].set_title(f"{name1} Delaunay triangulation with avg points")
                # axs[0].imshow(im1)
                # axs[0].scatter(vec1[:, 0], vec1[:, 1])
                # axs[0].triplot(vec1[:, 0], vec1[:, 1], tri.simplices)
                # axs[1].set_title(f"{name2} Delaunay triangulation with avg points")
                # axs[1].imshow(im1)
                # vec2 = avg_pts
                # axs[1].scatter(vec2[:, 0], vec2[:, 1])
                # axs[1].triplot(vec2[:, 0], vec2[:, 1], tri.simplices)
                # plt.show()
                # threads[fn] = executor.submit(morph_shape, im1, np.array(pts1), avg_pts, tri)
                threads[fn] = executor.submit(morph, im1, avg_img, np.array(pts1), avg_pts, tri, 0.0, 0.8)
        results = {k: thread.result() for k, thread in tqdm(threads.items(), "Getting Results")}
        new_avg = results
    # for fn, data in tqdm(dataset.items(), "Making avg face morphs"):
    #     pts1 = data.get('pts')
    #     im1 = data.get('img')
    #     if pts1 is not None and im1 is not None:
    #         tri = compute_tri_with_array(pts1, avg_pts)
    #         new_avg[fn] = morph_shape(im1, pts1, avg_pts, tri)
    return new_avg
def graph_with_avg_morph(im1, im2, pts1, pts2, im1name, im2name):
    im1pts = pts1
    img1 = im1
    
    full_morph1 = morph(img1, im2, im1pts, pts2, Delaunay(((pts2))), 0.0, 0.8)
    hybrid_morph1 = morph(img1, im2, im1pts, pts2, Delaunay(((pts2 + pts1) / 2)), 0.5, 0.5)
    f, axs = plt.subplots(2,2, figsize=(8, 6))
    axs[0,0].set_title(f"{im1name}:")
    axs[0,0].imshow((img1 * 255).astype(np.uint8))
    axs[0,1].set_title(f"{im2name}:")
    axs[0,1].imshow(im2)
    axs[1,0].set_title(f"{im1name} full morph to {im2name}")
    axs[1,0].imshow(full_morph1)
    axs[1,1].set_title(f"Hybrid morph {im1name} to {im2name}")
    axs[1,1].imshow(hybrid_morph1)
    plt.tight_layout()
    plt.show()
    return full_morph1, hybrid_morph1
def caricature(pts1: np.array, pts2: np.array, mag: float= 1.2) -> np.array:
    """Given pts1 P and pts2 Q, compute caricature pts via extrapolation as
    MAG * (P - Q) + Q

    Args:
        pts1 (np.array): points P
        pts2 (np.array): points Q
        mag (float, optional): magnitude to multiply diff of features by. Defaults to 1.2.

    Returns:
        np.array: caricature landmarks
    """
    diff = pts1 - pts2
    extrap_pts = mag * diff
    return pts2 + extrap_pts
def main():
    img_folder = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "images")
    json_path = osp.join(img_folder, "aligned_IMG_0991_aligned_DerekPicture(2).json")
    
    
    
    #mid face
    if MIDFACE:
        annotations = get_correspondence_pts_from_JSON(json_path)
    
        img1 = skimage.io.imread(osp.join(img_folder, "aligned_IMG_0991.JPG"))
        img1 = skimage.img_as_float(img1)
        img2 = skimage.io.imread(osp.join(img_folder, "aligned_DerekPicture.jpg"))
        img2 = skimage.img_as_float(img2)
        
        pts1: list[list[int]] = annotations.get("im1Points")
        pts2: list[list[int]] = annotations.get("im2Points")
        
        
        tri: Delaunay = compute_triangles(pts1, pts2)
        
        vec1 = np.array(pts1)
        vec2 = np.array(pts2)
        f, axs = plt.subplots(1, 2)
        name1 = annotations.get("im1_name")
        name2 = annotations.get("im2_name")
        
        axs[0].set_title(f"{name1} Delaunay triangulation with avg points")
        axs[0].imshow(img1)
        axs[0].scatter(vec1[:, 0], vec1[:, 1])
        axs[0].triplot(vec1[:, 0], vec1[:, 1], tri.simplices)
        axs[1].set_title(f"{name2} Delaunay triangulation with avg points")
        axs[1].imshow(img2)
        axs[1].scatter(vec2[:, 0], vec2[:, 1])
        axs[1].triplot(vec2[:, 0], vec2[:, 1], tri.simplices)
        plt.show()
        warp_frac: float = 0.5
        dissolve_frac: float = 0.5
        morphed_img = compute_mid_face(img1, img2, vec1, vec2, tri)
        
        f, axs = plt.subplots(1, 3)
        axs[0].imshow(morphed_img)
        
        axs[0].set_title("Hybrid Image")
        axs[1].imshow(img1)
        axs[1].set_title("IMG 1")
        axs[2].imshow(img2)
        axs[2].set_title("IMG 2")
        plt.tight_layout()
        plt.show()
    
    #morph
    if MORPH:
        fps = 30.0
        duration = 10.0
        out_path = osp.join(img_folder, "morph.mp4")
        frames = make_morph_video(img1, img2, vec1, vec2, tri, duration, fps)
        create_cv2_video(frames, out_path, fps, debug=False)
        
    #mean face
    if MEANFACE:
        img_dataset = osp.join(img_folder, "imm_face_db")
        pts_dataset = osp.join(img_folder, "imm_face_db")
        # img_dataset = osp.join(img_folder, "dataset")
        # pts_dataset = osp.join(img_folder, "dataset_pts")
        dest_folder = osp.join(img_folder, "avg_faces")
        if not osp.exists(dest_folder):
            os.makedirs(dest_folder)
        dataset = load_asf_dataset(img_dataset, pts_dataset)
        # for k, v in dataset.items():
        #     for img_name, img in v.items():
        #         plt.imshow(img)
        #         plt.show()
        # dataset = load_dataset(img_dataset, pts_dataset)
        avg_face_pts, avg_img = compute_avg_key_pts(dataset, get_img=False)
        new_avg = turn_dataset_into_avg(dataset, avg_face_pts, avg_img)
        side = np.round(np.sqrt(len(new_avg))).astype(int)
        other_side = side
        if side * other_side < len(new_avg):
            other_side += 1
        f, axs = plt.subplots(side, other_side, figsize=(16,11))
        queue = list(new_avg.items())
        avg_face = avg_img
        for r in tqdm(range(side), "Making plot"):
            for c in range(other_side):
                if len(queue) <= 0:
                    avg_face = np.round(np.mean(list(new_avg.values()), axis=0)).astype(np.uint8)
                    axs[r, c].set_title("Average face")
                    plt.imsave(osp.join(img_folder, "avg_face.jpg"), avg_face)
                    axs[r, c].imshow(avg_face)
                    break
                else:
                    next_item = queue.pop(0)
                    # axs[r, c].set_title(next_item[0][:-3])
                    axs[r, c].imshow(next_item[1])
                    fn = osp.join(dest_folder, f"morphed_to_mean_shape_{next_item[0]}")
                    plt.imsave(fn, next_item[1])
                    
        # plt.tight_layout()
        plt.show()
        queue = list(new_avg.items())
        # plt.close()
        test1 = morph(dataset[queue[22][0]].get('img'), avg_face / 255, np.array(dataset[queue[22][0]].get('pts')), avg_face_pts, Delaunay(avg_face_pts), 0.5, 0.5)
        f, axs = plt.subplots(1,3)
        axs[0].set_title(queue[22][0])
        axs[0].imshow((dataset[queue[22][0]].get('img') * 255).astype(np.uint8))
        axs[1].set_title(f"morph {queue[22][0]} mid-way to avg face")
        axs[1].imshow(test1)
        axs[2].set_title("avg_face")
        axs[2].imshow(avg_face)
        plt.tight_layout()
        plt.show()
        f, axs = plt.subplots(side, other_side, figsize=(16,11))
        to_avg = turn_dataset_into_avg(dataset, avg_face_pts, avg_face / 255, Delaunay(avg_face_pts), warp_frac=0.5, dissolve_frac=0.5)
        queue = list(to_avg.items())
        avg_face = avg_img
        side = np.round(np.sqrt(len(to_avg))).astype(int)
        other_side = side
        for r in tqdm(range(side), "Making plot"):
            for c in range(other_side):
                if len(queue) <= 0:
                    avg_face = np.round(np.mean(list(new_avg.values()), axis=0)).astype(np.uint8)
                    axs[r, c].set_title("Average face")
                    # plt.imsave(osp.join(img_folder, "avg_face.jpg"), avg_face)
                    axs[r, c].imshow(avg_face)
                    break
                else:
                    next_item = queue.pop(0)
                    axs[r, c].imshow(next_item[1])
                    fn = osp.join(dest_folder, f"morphed_to_avg_face_{next_item[0]}")
                    plt.imsave(fn, next_item[1])
        plt.show()
        json_path = osp.join(img_folder, "aligned_me_cropped_to_crop_to(3).json")
        img1 = skimage.io.imread(osp.join(img_folder, "aligned_me_cropped.jpg")) / 255
        annotations = get_correspondence_pts_from_JSON(json_path)
        height, width = img1.shape[:2]
        corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
        im1pts = np.array(corners + annotations.get("im1Points"))
        im1name = annotations.get('im1_name')
        avg_name = "avg_face.jpg"
        f, axs = plt.subplots(1, 2)
        im1 = img1
        pts1 = im1pts
        avg_pts = avg_face_pts
        name1 = ("im1_name")
        name2 = ("im2_name")
        vec1 = im1pts
        axs[0].set_title(f"{name1} Delaunay triangulation with avg points")
        axs[0].imshow(im1)
        axs[0].scatter(vec1[:, 0], vec1[:, 1])
        tri = Delaunay(vec1)
        axs[0].triplot(vec1[:, 0], vec1[:, 1], tri.simplices)
        axs[1].set_title(f"{name2} Delaunay triangulation with avg points")
        axs[1].imshow(avg_face)
        vec2 = avg_face_pts
        tri = Delaunay(avg_pts)
        axs[1].scatter(vec2[:, 0], vec2[:, 1])
        axs[1].triplot(vec2[:, 0], vec2[:, 1], tri.simplices)
        plt.show()
        assert im1name is not None
        full_morph, hybrid_morph = graph_with_avg_morph(img1, avg_face / 255, im1pts, avg_face_pts, im1name, avg_name)
        fm1 = osp.join(img_folder, f"full_morph_{im1name[:-4]}_to_{avg_name[:-4]}.jpg")
        hm1 = osp.join(img_folder, f"hybrid_morph_{im1name[:-4]}_to_{avg_name[:-4]}.jpg")
        plt.imsave(fm1, full_morph)
        plt.imsave(hm1, hybrid_morph)
        
        full_morph, hybrid_morph = graph_with_avg_morph(avg_face / 255, img1, avg_face_pts, im1pts, avg_name, im1name)
        fm1 = osp.join(img_folder, f"full_morph_{avg_name[:-4]}_to_{im1name[:-4]}.jpg")
        hm1 = osp.join(img_folder, f"hybrid_morph_{avg_name[:-4]}_to_{im1name[:-4]}.jpg")
        plt.imsave(fm1, full_morph)
        plt.imsave(hm1, hybrid_morph)
    if CARICATURE:
        if not MEANFACE:
            img_dataset = osp.join(img_folder, "imm_face_db")
            pts_dataset = osp.join(img_folder, "imm_face_db")
            # img_dataset = osp.join(img_folder, "dataset")
            # pts_dataset = osp.join(img_folder, "dataset_pts")
            dest_folder = osp.join(img_folder, "avg_faces")
            if not osp.exists(dest_folder):
                os.makedirs(dest_folder)
            dataset = load_asf_dataset(img_dataset, pts_dataset)
            # for k, v in dataset.items():
            #     for img_name, img in v.items():
            #         plt.imshow(img)
            #         plt.show()
            # dataset = load_dataset(img_dataset, pts_dataset)
            avg_face_pts, avg_img = compute_avg_key_pts(dataset, get_img=False)
            new_avg = turn_dataset_into_avg(dataset, avg_face_pts, avg_img)
        json_path = osp.join(img_folder, "aligned_me_cropped_to_crop_to(3).json")
        img1 = skimage.io.imread(osp.join(img_folder, "aligned_me_cropped.jpg")) / 255   
        annotations = get_correspondence_pts_from_JSON(json_path)
        height, width = img1.shape[:2]
        corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
        im1pts = np.array(corners + annotations.get("im1Points"))
        caric_pts = caricature(im1pts, avg_face_pts, mag=1.5)
        # caric_img = morph_shape(img1, im1pts, caric_pts, Delaunay(caric_pts))
        im1name = annotations.get('im1_name')
        avg_name = "avg_face.jpg"
        caric_full, caric_half = graph_with_avg_morph(img1, avg_face / 255, im1pts, caric_pts, im1name, avg_name)
        plt.imsave(osp.join(img_folder, "caric_full.jpg"), caric_full)
        plt.imsave(osp.join(img_folder, "caric_half.jpg"), caric_half)
    
        
    return
                    
            
        
        
        
        
if __name__ == "__main__":
    main()
    
    
    