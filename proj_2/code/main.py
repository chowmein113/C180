import numpy as np
import skimage as sk
from skimage.transform import resize, rescale
import skimage.io as skio

from skimage import feature
from scipy.signal import convolve2d
from scipy.ndimage import shift

from concurrent.futures.thread import ThreadPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import os.path as osp
import cv2

def finite_diff_op():
    Dx = np.array([[1, -1]])
    Dy = np.array([[1, -1]]).T
    parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    filename = osp.join(parent_dir,"images")
    filename = osp.join(filename, "cameraman.png")
    img = skio.imread(filename)
    alpha = img[:,:,3]
    rgb = img[:,:,:3]
    greyscale = sk.color.rgb2gray(rgb)
    im = sk.img_as_float(greyscale)
    thresh = 0.35
    d_x = convolve2d(im, Dx, mode='same', boundary='symm')
    d_y = convolve2d(im, Dy, mode='same', boundary='symm')
    
    gm = np.sqrt((d_x ** 2) + (d_y ** 2))
    gm_filtered = gm > thresh
    
    std = 1
    g = cv2.getGaussianKernel(ksize=int(1 + 6 * std), sigma=std)
    g2d = g @ g.T
    
    lpf_img = convolve2d(im, g2d, mode='same', boundary='symm')
    lpf_dx = convolve2d(lpf_img, Dx, mode='same', boundary='symm')
    lpf_dy = convolve2d(lpf_img, Dy, mode='same', boundary='symm')
    gm_lpf_img = np.sqrt(lpf_dx ** 2 + lpf_dy** 2)
    dgx = convolve2d(g2d, Dx, mode='same', boundary='symm')
    dgy = convolve2d(g2d, Dy, mode='same', boundary='symm')
    
    dg_x = convolve2d(im, dgx, mode='same', boundary='symm')
    dg_y = convolve2d(im, dgy, mode='same', boundary='symm')
    
    gmdg = np.sqrt(dg_x ** 2 + dg_y ** 2)
    f, axs = plt.subplots(2,2)
    axs[0, 0].imshow(im)
    axs[0, 0].set_title('original')
    axs[1, 0].set_title('Dx')
    axs[1, 0].imshow(d_x)
    axs[0, 1].set_title('Dy')
    axs[0, 1].imshow(d_y)
    axs[1, 1].set_title(f'Gradient Magnitude w/ threshold value: {thresh}')
    axs[1, 1].imshow(gm_filtered)
    plt.show()
    
    f, axs = plt.subplots(2,3)
    axs[0, 0].imshow(lpf_img)
    axs[0, 0].set_title('Low Pass Filtered Image')
    axs[1, 0].set_title('Dx of lpf image')
    axs[1, 0].imshow(lpf_dx)
    axs[0, 1].set_title('Dy of lpf image')
    axs[0, 1].imshow(lpf_dy)
    axs[1, 1].set_title('Gradient Magnitude of derivate of DoG filter applied to original im')
    axs[1, 1].imshow(gmdg)
    axs[0, 2].imshow(gm_lpf_img)
    axs[0, 2].set_title("Gradient Magnitude by applying d filters to blurred img")
    plt.tight_layout()
    plt.show()
    
    f, axs = plt.subplots(4,2)
    axs[0, 0].imshow(im)
    axs[0, 0].set_title('original')
    axs[1, 0].set_title('Dx')
    axs[1, 0].imshow(d_x)
    axs[0, 1].set_title('Dy')
    axs[0, 1].imshow(d_y)
    axs[1, 1].set_title('Gradient Magnitude')
    axs[1, 1].imshow(gm_filtered)
    
    axs[2, 0].imshow(lpf_img)
    axs[2, 0].set_title('Low Pass Filtered Image')
    axs[3, 0].set_title('Dx of lpf image')
    axs[3, 0].imshow(lpf_dx)
    axs[2, 1].set_title('Dy of lpf image')
    axs[2, 1].imshow(lpf_dy)
    axs[3, 1].set_title('Gradient Magnitude of derivate of DoG filter applied to original im')
    axs[3, 1].imshow(gmdg)
    
    plt.tight_layout()
    
    plt.show()
def main():
    finite_diff_op()

if __name__ == '__main__':
    main()
    