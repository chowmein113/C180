import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
import cv2
def get_image_as_float(filename):
    img = skio.imread(filename)
    r = sk.img_as_float(img[:, :, 0])
    g = sk.img_as_float(img[:, :, 1])
    b = sk.img_as_float(img[:, :, 2])
    return np.dstack((r, g, b))
def convert(img):
    """Clips values of array 

    Args:
        img (np.array): Numpy matrix rep of image

    Returns:
        np.array: clipped version of numpy matrix
    """
    # r_max  = np.max(img)
    # r_min = np.min(img)
    # final = (img + r_min) / r_max
    # final = (255 * final).astype(np.uint8)
    # final2 = np.clip(final, 0, 255)
    # return final2
    result: np.array = img
    if np.issubdtype(img.dtype, np.integer):
        result = np.clip(img, 0, 255)
    elif np.issubdtype(img.dtype, np.float64):
        result = np.clip(img, 0, 1)
    return result
def convert_img_to_float(img: np.array) -> np.array:
    """Converts IMG from int type to float type using sk utility

    Args:
        img (np.array): 3D or 2D numpy matrix of image

    Returns:
        np.array: float rep of image
    """
    result = np.copy(img).astype(np.float64)
    if np.issubdtype(img.dtype, np.integer):
        
        if len(img.shape) > 2:
            for i in range(img.shape[2]):
                result[:, :, i] = sk.img_as_float(img[:, :, i])
        else:
            result = sk.img_as_float(img)
    return result
def feather_mask(img: np.array, dist: int = 0, degree: float = 0.0) -> np.array:
    """creates a feather mask half way on img based on dist for blending and degree of rotation

    Args:
        img (np.array): np matrix to make mask for
        dist (int, optional): dist of linear change from 0 to 1 bwteen mask and img. Defaults to 0.
        degree (float, optional): degree of rotation of mask. Defaults to 0.0.

    Returns:
        np.array: img with mask
    """
    blend = np.copy(img)
    grad = np.linspace(0, 1, dist)
    height = img.shape[0]
    width = img.shape[1]
    half_width = img.shape[1] // 2
    half_height = img.shape[0] // 2
    filt = []
    for i in range(max(height, width)):
        filt.append(grad)
    filt = np.vstack(filt)
    
    #rotate elements according to degree
    rFilt = cv2.getRotationMatrix2D((half_width, half_height), degree, 1)
    
    filt_rotated = cv2.warpAffine(filt, rFilt, blend.shape[:2], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
    return filt_rotated
def multiply_mask_to_img(mask, img):
    img_res = np.zeros(img.shape)
    if len(img.shape) > 2:
        for i in range(img.shape[2]):
            img_res[:, :, i] = img[:, :, i] * mask
    else:
        img_res = mask * img
    return img_res
        
def normalize_matrix(img: np.array) -> np.array:
    """Take matrix and scale elements it by normalizing

    Args:
        img (np.array): np matrix

    Returns:
        np.array: IMG with normalized elements
    """
    mat_max = np.max(img)
    mat_min = np.min(img)
    max_dist = mat_max - mat_min
    if max_dist == 0:
        return (img - mat_min) * 0
    return (img - mat_min) / (max_dist)

###Outsourced
class Selector(object):
    
    
    def __init__(self, img):
        f, axs = plt.subplots()
        self.mask = None
        axs.imshow(img)
        axs.set_title("Draw to Select Region of Image")
        
        self.region = LassoSelector(axs, self.grab_region)
        
        self.img = np.copy(img)
        
        plt.show()

    
    def grab_region(self, pts):
        plt.close()
        empty_mask = np.zeros(self.img.shape[:2], dtype=self.img.dtype)
        vector = np.array(pts)
        print(f"pts: {pts}")
        print(f"vector: {vector.shape} \nvector min and max {np.min(vector), np.max(vector)}\n vector itself: {vector}")
        cv2.fillPoly(empty_mask, [vector.astype(int)], 1)
        filled_mask = empty_mask
        print(f"mask filled with shape {filled_mask.shape, empty_mask.shape} and mask points: {np.max(filled_mask), np.min(filled_mask)}")
        self.mask = filled_mask
        plt.imshow(filled_mask)
        plt.title("Selected Region")
        plt.show()
        
    def get_mask(self) -> np.array:
        return self.mask
    
    def get_mask_on_img(self) -> np.array:
        return self.img * self.mask
    