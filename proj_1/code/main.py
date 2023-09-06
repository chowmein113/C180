# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
from skimage.transform import resize, rescale
import skimage.io as skio
from skimage import feature
from scipy.signal import convolve2d
from scipy.ndimage import shift
from tools import gaussian_kernel
from concurrent.futures.thread import ThreadPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

RESCALE_FIRST = True
PARALLEL_PRO = True
DEBUG = True
def create_colored_image(filename: str, crop_percent: float, preprocess_func = lambda x: x, lvls = 0, debug = False) -> np.array:
    img = skio.imread(filename)
    im = sk.img_as_float(img)
    
    height = np.floor(im.shape[0] / 3.0).astype(np.int64)
    width = im.shape[1]
    
    b = im[:height]
    g = im[height: 2 * height]
    r = im[2 * height: 3 * height]
    
    #preprocess images to edge cased pictures
    ##border crop
    if debug:
        hcrop: int = int(height * (crop_percent / 100))
        wcrop: int = int(width * (crop_percent / 100))
        bc = b[hcrop : height - hcrop, wcrop : width - wcrop]
        gc = g[hcrop : height - hcrop, wcrop : width - wcrop]
        rc = r[hcrop : height - hcrop, wcrop : width - wcrop]
        
    
        f, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(rc)
        ax[0, 1].imshow(gc)
        ax[1, 0].imshow(bc)
        ax[1, 1].imshow(rc + gc + bc)
        plt.show()
    bc = b
    gc = g
    rc = r
    
    #process image
    bp = preprocess_func(bc).astype(np.float64)
    gp = preprocess_func(gc).astype(np.float64)
    rp = preprocess_func(rc).astype(np.float64)
    
    if debug:
        f, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(rp)
        ax[0, 1].imshow(gp)
        ax[1, 0].imshow(bp)
        ax[1, 1].imshow(rp + gp + bp)
    
    #calculate best translations
    best_trans = pyramidgauss(rp, gp, bp, crop_percent, lvls=lvls)
    best_r_trans = best_trans[0]
    best_g_trans = best_trans[1]
    
    #calculate result
    non_trans = np.zeros((height, width, 3))
    non_trans[:, :, 0] = r
    non_trans[:, :, 1] = g
    non_trans[:, :, 2] = b
    result = np.zeros((height, width, 3))
    r = shift2(r, (best_r_trans[0], best_r_trans[1]))
    result[:, :, 0] = r
    g = shift2(g, (best_g_trans[0], best_g_trans[1]))
    result[:, :, 1] = g
    result[:, :, 2] = b
    result = (255 * result).astype(np.uint8)
    
    return result, (255 * non_trans).astype(np.uint8), best_r_trans, best_g_trans
    
    
    
     ##align the images
     
def pyramidgauss(r: np.array, g: np.array, b: np.array, crop_percent = 0, lvls=0) -> list[list[int]]:
    """This function uses the idea of pyramid scaling to find translation on 
    the smallest coursest version of image which makes the search time very small.
    Once building back up to true size, only need to search with displacement range of
    ~2 pixels

    Args:
        r (np.array): r_channel 2d matrix of img
        g (np.array): g_channel 2d matrix of img
        b (np.array): b_channel 2d matrix of img

    Returns:
        list[list[int]]: best_r_trans, best_g_trans
    """
    queue = []
    height = r.shape[0]
    width = r.shape[1]
    
    #set max pixel displacement to search for, empirical value
    displacement = 15
    
    best_r_trans = [0, 0]
    best_g_trans = [0, 0]
    
    
    if not RESCALE_FIRST:
        queue.append([height, width])
        if lvls == 0:
            while height > 100 and width > 100:
                height //= 2
                width //= 2
                
                queue.append([height, width])
        elif lvls > 1:
            for lvl in range(0, lvls):
                height //= 2
                width //= 2
                queue.append([height, width])
        displacement = (width // 15)
        
        
    else:
        p_rgb = [r, b, g]
        queue.append(p_rgb)
        
        if lvls == 0:
            lvl = 1
            while width > 100:
                scale_factor = 1 / (2 ** lvl)
                width = int(scale_factor * width)
                lvl += 1
                with ThreadPoolExecutor(max_workers=3) as executor:
                
                    scaled_r = executor.submit(rescale, r, scale_factor, anti_aliasing=False)
                    scaled_b = executor.submit(rescale, b, scale_factor, anti_aliasing=False)
                    scaled_g = executor.submit(rescale, g, scale_factor, anti_aliasing=False)
                    p_rgb[0] = scaled_r.result()
                    p_rgb[1] = scaled_b.result()
                    p_rgb[2] = scaled_g.result()
                queue.append(p_rgb)
        elif lvls > 1:
            for lvl in range(1, lvls):
                scale_factor = 1 / (2 ** lvl)
                next_size = [height, width]
                with ThreadPoolExecutor(max_workers=3) as executor:
                
                    scaled_r = executor.submit(rescale, r, scale_factor, anti_aliasing=False)
                    scaled_b = executor.submit(rescale, b, scale_factor, anti_aliasing=False)
                    scaled_g = executor.submit(rescale, g, scale_factor, anti_aliasing=False)
                    p_rgb[0] = scaled_r.result()
                    p_rgb[1] = scaled_b.result()
                    p_rgb[2] = scaled_g.result()
                queue.append(p_rgb)
    
    queue.reverse()
        
    
    for next_size in tqdm(queue, "going through pyramid level: "):
        # next_size = queue.pop(-1)
        #find best alignment
        # if displacement == 15:
        #     scaled_r = rescale(r, next_size, anti_aliasing=False)
        #     scaled_b = rescale(b, next_size, anti_aliasing=False)
        #     scaled_g = rescale(g, next_size, anti_aliasing=False)
        # else:
        #     scaled_r = resize(r, next_size, anti_aliasing=False)
        #     scaled_b = resize(b, next_size, anti_aliasing=False)
        #     scaled_g = resize(g, next_size, anti_aliasing=False)
        #original size
        
        if RESCALE_FIRST:
            p_rgb = next_size
            scaled_r = p_rgb[0]
            scaled_b = p_rgb[1]
            scaled_g = p_rgb[2]
        elif r.shape[0] == next_size[0] and r.shape[1] == next_size[1]:
            scaled_r = r
            scaled_b = b
            scaled_g = g
        #first pass and coursest image and downsized
        
        else:
            scaled_r = resize(r, next_size, anti_aliasing=False)
            scaled_b = resize(b, next_size, anti_aliasing=False)
            scaled_g = resize(g, next_size, anti_aliasing=False)
            
        height = scaled_r.shape[0]
        width = scaled_r.shape[1]
        #crop border each time in pyramid now
        hcrop: int = int(height * (crop_percent / 100))
        wcrop: int = int(width * (crop_percent / 100))
        scaled_r = scaled_r[hcrop : height - hcrop, wcrop : width - wcrop]
        scaled_b = scaled_b[hcrop : height - hcrop, wcrop : width - wcrop]
        scaled_g = scaled_g[hcrop : height - hcrop, wcrop : width - wcrop]
        
        if displacement == 15:
            displacement = scaled_b.shape[0] // 15 if scaled_b.shape[0] > scaled_b.shape[1] else scaled_b.shape[1] // 15
        if PARALLEL_PRO:
            prev_r = best_r_trans
            prev_g = best_g_trans
            best_r_trans = parallel_align(scaled_r, scaled_b, displacement, [2 * i for i in best_r_trans])
            best_g_trans = parallel_align(scaled_g, scaled_b, displacement, [2 * i for i in best_g_trans])
        else:
            prev_r = best_r_trans
            prev_g = best_g_trans
            best_r_trans = simple_align(scaled_r, scaled_b, displacement, [2 * i for i in best_r_trans])
            best_g_trans = simple_align(scaled_g, scaled_b, displacement, [2 * i for i in best_g_trans])
        displacement = 2
    return [best_r_trans, best_g_trans]
        
        
def shift2(matrix, shift_values):
    """given matrix and shift values, shift matrix and return shifted matrix

    Args:
        matrix (2D np.array): 2D img matrix
        shift_values (list or tuple): shift deltas in form (dx, dy)

    Returns:
        _type_: _description_
    """
    return shift(matrix, shift_values, mode='constant', cval=0) 
    # return np.roll(matrix, (shift_values[0], shift_values[1]), axis=(1, 0))   

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
def parallel_align(im_1, im_2, displacement: int, start_trans: list[int] = [0, 0]) -> list[int]:
    """Aligns images via Normalized Cross-Correlation w/ multithreading; good for large computations

    Args:
        im_1 (np.array): 2D numpy array w/ first image as base
        im_2 (np.array): 2D numpy array w/ second image comparing to base
        displacement (int): number of pixel displacements to search up until
    
    Returns:
        list[int]: best translation to move im_2 to im_1
    """
    assert im_1.shape[0] == im_2.shape[0]
    assert im_1.shape[1] == im_2.shape[1]

    def get_ncc(dx, dy):
        
        vec1 = shift2(im_1, (start_trans[0] + dx, start_trans[1] + dy)).flatten()
        ncc_val = ncc(vec1, vec2)
        
        return ncc_val, [start_trans[0] + dx, start_trans[1] + dy]
    # base_avg = np.mean(im_2)
    # comp_avg = np.mean(im_1)
    # vec1 = im_1.flatten()
    vec2 = im_2.flatten()
    ddist: list[int] = list(range(-displacement, displacement + 1))
    max_val = -(float('inf'))
    best_trans = start_trans[:]
    with ThreadPoolExecutor(max_workers=7) as executor:
        results = []
        for diff_x in ddist:
            for diff_y in ddist:
                results.append(executor.submit(get_ncc, diff_x, diff_y))

        for result in tqdm(results, "Aligning w/ max displacement: {}".format(displacement)):
            ncc_val, trans = result.result()
            # if ncc_val > max_val or (np.isclose(ncc_val, max_val) 
            #                          and (abs(trans[0]) > abs(best_trans[0])
            #                               or (abs(trans[1])) > abs(best_trans[1]))):
            if ncc_val > max_val:
                best_trans[0] = trans[0]
                best_trans[1] = trans[1]
                max_val = ncc_val
    return best_trans
def edge_and_color(img: np.array) -> np.array:
    img_weighted = img + feature.canny(img, sigma=3)
    return img_weighted


def get_edge_image(img: np.array) -> np.array:
    
    gauss_img = apply_gaussian_blur(img)
    #canny thresholding technique
    # for row in range(gauss_img.shape[0]):
    #     for col in range(gauss_img.shape[1]):
    #         val = gauss_img[row][col]
    #         if val < 0.1:
    #             gauss_img[row][col] = 0.0
    #         elif 0.1 <= val and val <= 0.4:
    #             gauss_img[row][col] = 0.5
    #         else:
    #             gauss_img[row][col] = 1.0
            
    return gauss_img - img

def apply_gaussian_blur(img: np.array) -> np.array:
    """Applies gaussian blur based on gaussian kernel with scipy
    convolve2D

    Args:
        img (np.array):2D greyscale img to be given gaussian blur img

    Returns:
        np.array: blurred img with gaussian blur
    """
    gauss = gaussian_kernel
    result = convolve2d(img, gauss, mode='same', boundary='wrap')
    return result
        
def ssd(vec1, vec2):
    return -np.sum((vec1 - vec2) ** 2)

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
    
def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # name of the input file
    data = os.path.join(root, "data")
    imname = 'monastery.jpg'
    imname = input("name of file: ")
    lvls = int(input("How many levels for pyramid? 1=no pyramid, 0=automatic division by 2 until img less that 100 x 100 pixels: "))
    CROP_PERCENT = 12.5
    FUNC_METHOD = "edge and color"
    img_path = os.path.join(data, imname)
    # use skleans uncanny
    img, non_trans, best_r_trans, best_g_trans = create_colored_image(img_path, CROP_PERCENT, lambda img: feature.canny(img, sigma=6), lvls=lvls, debug=DEBUG)
    #hand crafted uncanny or gaussian blur subtraction:look in function
    # img, non_trans, best_r_trans, best_g_trans = create_colored_image(img_path, CROP_PERCENT, get_edge_image, lvls=lvls, debug=DEBUG)
    # color based aligning
    # img, non_trans, best_r_trans, best_g_trans = create_colored_image(img_path, CROP_PERCENT, lambda x: x, debug = DEBUG)
    #canny filter plus img color to add more weight to edges but also look for color similarity
    # img, non_trans, best_r_trans, best_g_trans = create_colored_image(img_path, CROP_PERCENT, edge_and_color, lvls=lvls, debug=DEBUG)
    # save the image
    copies = 0
    
    fname = 'out_aligned_{}_crop_{}_method_{}_red_x_{}_y_{}_green_x_{}_y_{}_{}'.format(copies, CROP_PERCENT, FUNC_METHOD, best_r_trans[0], best_r_trans[1], best_g_trans[0], best_g_trans[1], imname)
    faname = 'out_not_aligned_{}_crop_{}_method_{}_red_x_{}_y_{}_green_x_{}_y_{}_{}'.format(copies, CROP_PERCENT, FUNC_METHOD, best_r_trans[0], best_r_trans[1], best_g_trans[0], best_g_trans[1], imname)
    
    fname = os.path.join(root, 'images', fname)
    faname = os.path.join(root, 'images', faname)
    while (os.path.exists(faname)):
        copies += 1
        fname = 'out_aligned_{}_crop_{}_method_{}_red_x_{}_y_{}_green_x_{}_y_{}_{}'.format(copies, CROP_PERCENT, FUNC_METHOD, best_r_trans[0], best_r_trans[1], best_g_trans[0], best_g_trans[1], imname)
        faname = 'out_not_aligned_{}_crop_{}_method_{}_red_x_{}_y_{}_green_x_{}_y_{}_{}'.format(copies, CROP_PERCENT, FUNC_METHOD, best_r_trans[0], best_r_trans[1], best_g_trans[0], best_g_trans[1], imname)
        fname = os.path.join(root, 'images', fname)
        faname = os.path.join(root, 'images', faname)
    
    # display the image
    f, axs = plt.subplots(2, 1)
    axs[0].imshow(img)
    axs[0].set_title("combined img")
    axs[1].imshow(non_trans)
    axs[1].set_title("placed on top of each other (no alignment)")
    plt.show()
    
    skio.imsave(fname, img)
    skio.imsave(faname, non_trans)

    # display the image

    return
if __name__ == '__main__':
    main()