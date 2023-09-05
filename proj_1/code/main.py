# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
from skimage.transform import resize
import skimage.io as skio
from scipy.signal import convolve2d
from tools import gaussian_kernel
from concurrent.futures.thread import ThreadPoolExecutor
import os


def create_colored_image(filename: str, crop_percent: float, preprocess_func = lambda x: x) -> np.array:
    img = skio.imread(filename)
    im = sk.img_as_float(img)
    
    height = np.floor(im.shape[0] / 3.0).astype(np.int)
    width = im.shape[1]
    
    b = im[:height]
    g = im[height: 2 * height]
    r = im[2 * height: 3 * height]
    
    #preprocess images to edge cased pictures
    ##border crop
    hcrop = height * (crop_percent / 100)
    wcrop = width * (crop_percent / 100)
    b = b[hcrop : height - hcrop, wcrop : width - wcrop]
    g = g[hcrop : height - hcrop, wcrop : width - wcrop]
    r = r[hcrop : height - hcrop, wcrop : width - wcrop]
    
    #process image
    bp = preprocess_func(b)
    gp = preprocess_func(g)
    rp = preprocess_func(r)
    
    #calculate best translations
    best_trans = pyramidgauss(rp, gp, bp)
    best_r_trans = best_trans[0]
    best_g_trans = best_trans[1]
    
    #calculate result
    result = np.zeros((height, width, 3))
    r = np.roll(r, (best_r_trans[0], best_r_trans[1]), axis=(0, 1))
    result[:, :, 1] = r
    g = np.roll(g, (best_g_trans[0], best_g_trans[1]), axis=(0, 1))
    result[:, :, 2] = g
    result[:, :, 3] = b
    
    return result
    
    
    
     ##align the images
     
def pyramidgauss(r: np.array, g: np.array, b: np.array) -> list[list[int]]:
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
    while height > 125 and width > 125:
        queue.append([height, width])
        height /= 2
        width /= 2
    while len(queue) > 0:
        next_size = queue.pop(-1)
        #find best alignment
        scaled_r = resize(r, next_size, mode='reflect', anti_aliasing=True)
        scaled_b = resize(b, next_size, mode='reflect', anti_aliasing=True)
        scaled_g = resize(g, next_size, mode='reflect', anti_aliasing=True)
        best_r_trans = simple_align(scaled_r, scaled_b, displacement, best_r_trans)
        best_g_trans = simple_align(scaled_g, scaled_b, displacement, best_g_trans)
        displacement = 2
    return [best_r_trans, best_g_trans]
        
        
     
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
    ddist: list[int] = list(range(-displacement, displacement))
    max_val: int = int(- float('INF'))
    best_trans = start_trans
    for diff_x in ddist:
        for diff_y in ddist:
            vec1 = np.roll(im_2, (start_trans[0] + diff_x, start_trans[1] + diff_y), axis=(0, 1)).flatten()
            ncc_val = ncc(vec1, vec2)
            if ncc_val > max_val:
                best_trans[0] = diff_x
                best_trans[1] = diff_y
                max_val = ncc_val
    return best_trans

def get_edge_image(img: np.array) -> np.array:
    
    gauss_img = apply_gaussian_blur(img)
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
        
        

def ncc(vec1, vec2):
    """Compute Normalized Cross-Correlation between two vectors

    Args:
        vec1 (_type_): vector of image 1
        vec2 (_type_): vector of image 2
        vec1.shape = vec2.shape

    Returns:
        int: value of dot product of normalized vectors
    """
    f_norm1 = np.linalg.norm(vec1)
    f_norm2 = np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (f_norm1 * f_norm2)
    
def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # name of the input file
    data = os.path.join(root, "data")
    imname = 'cathedral.jpg'
    img_path = os.path.join(data, imname)
    
    img = create_colored_image(img_path, 15, get_edge_image)

    # save the image
    fname = 'out_{}'.format(imname)
    fname = os.path.join(root, 'images', fname)
    skio.imsave(fname, img)

    # display the image
    skio.imshow(img)
    skio.show()
if __name__ == '__main__':
    main()