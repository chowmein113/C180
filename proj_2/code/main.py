from flask import app
import numpy as np
import skimage as sk
from skimage.transform import resize, rescale
import skimage.io as skio

from skimage import feature
from scipy.signal import convolve2d
from scipy.ndimage import shift
from align_image_code import align_images
from concurrent.futures.thread import ThreadPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import os.path as osp
import cv2
from tools import get_image_as_float, convert, convert_img_to_float, normalize_matrix, feather_mask, Selector, multiply_mask_to_img
from PIL import Image

MAX_WORKERS = 7
def finite_diff_op():
    Dx = np.array([[1, -1]])
    Dy = np.array([[1, -1]]).T
    parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    fileroot = osp.join(parent_dir,"images")
    filename = osp.join(fileroot, "cameraman.png")
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
    
    std = 1.0
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
    
def unsharpen_mask_filter(filename, std, save=False, save_dest = ""):
    ### r = im[...,0]
    g = cv2.getGaussianKernel(ksize=int(1 + 6 * std), sigma=std)
    g2d = g @ g.T
    og_img = skio.imread(filename) if type(filename) == str else np.copy(filename)
    # f, axs = plt.subplots(2,2)
    img = skio.imread(filename) if type(filename) == str else filename
    if img.dtype != np.float64:
        r = sk.img_as_float(img[:, :, 0])
        g = sk.img_as_float(img[:, :, 1])
        b = sk.img_as_float(img[:, :, 2])
    else:
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
    e = np.zeros(g2d.shape)
    e[g2d.shape[0] // 2, g2d.shape[0] // 2] = 1
    
    mag = 3.5
    convol_mat = ((1 + mag) * e - mag * g2d)
    rg = convolve2d(r, g2d, mode='same', boundary='symm')
    gg = convolve2d(g, g2d, mode='same', boundary='symm')
    bg = convolve2d(b, g2d, mode='same', boundary='symm')
    low_pass_g = np.dstack((rg, gg, bg))
   
    rg = convolve2d(r, convol_mat, mode='same', boundary='symm')
    gg = convolve2d(g, convol_mat, mode='same', boundary='symm')
    bg = convolve2d(b, convol_mat, mode='same', boundary='symm')
    # axs[0, 0].imshow(rg)
    # axs[0, 0].set_title('red sharpened')
    # axs[0, 1].imshow(gg)
    # axs[0, 1].set_title('green sharpened')
    # axs[1, 0].imshow(bg)
    # axs[1, 0].set_title('blue sharpened')
    final = np.dstack((rg, gg, bg))
    # axs[1, 1].imshow(final)
    # axs[1, 1].set_title("put together")
    # plt.tight_layout()
    # plt.show()
    hpf_taj_r = convert(rg)
    
    
    hpf_taj_g = convert(gg)
    
    
    hpf_taj_b = convert(bg)
    
    
    # img[:, :, 0] = hpf_taj_r
    # img[:, :, 1] = hpf_taj_g
    # img[:, :, 2] = hpf_taj_b
    img_return = np.dstack((hpf_taj_r, hpf_taj_g, hpf_taj_b))
    img = final
    # img_return = final.astype(np.uint8)
    
    
    
    f, axs = plt.subplots(2,2)
    axs[0, 1].set_title(f"High Frequency Sharpen edges with magnitude alpha={mag}")
    axs[0, 1].imshow(img)
    axs[1, 1].set_title("Original Image")
    axs[1, 1].imshow(og_img)
    axs[0, 0].imshow(low_pass_g)
    axs[0, 0].set_title("low pass filter")
    axs[1, 0].imshow(img - low_pass_g)
    axs[1, 0].set_title("High Pass Filter")
    plt.tight_layout()
    plt.show()
    if save:
        plt.imsave(save_dest, img_return)
    return img
def view_frequency_bode_graph(grey_img: np.array) -> np.array:
    ft_graph: np.array = np.log(np.abs(np.fft.fftshift(np.fft.fft2(grey_img))))
    return ft_graph

def hybrid_image(img1: np.array, img2: np.array, \
    low_freq_cut: float = 1.0, \
        high_freq_cut: float = 1.0, display=False) -> np.array:
    """Given 2 aligned images, IMG1 AND IMG2, generate a low pass image on IMG1
    based on LOW_FREQ_CUT cutoff frequency and generate a high pass image on 
    IMG2 based on HIGH_FREQ_CUT cutoff frequency and stack them together.

    Args:
        img1 (np.array): greyscaled image matrix that is aligned with IMG2
        img2 (np.array): greyscaled image matrix that is aligned with IMG1

    Returns:
        np.array: the resulting layered hybrid image of IMG1 and IMG2
    """
    low_pass_img: np.array = blur(img1, gauss_kernel=None, std=low_freq_cut)
    high_pass_img: np.array = sharpen(img2, sigma=high_freq_cut)
    
    final: np.array = (low_pass_img + high_pass_img) / 2
    if final.dtype == np.float64:
        final = np.clip(final, 0, 1)
    if final.dtype == np.uint8:
        final = np.clip(final, 0, 255)
        
    if display:
        f, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(low_pass_img)
        axs[0, 0].set_title(f"Low pass image with low frequency cutoff val: {low_freq_cut}")
        
        axs[0, 1].imshow(high_pass_img)
        axs[0, 1].set_title(f"High pass image with high frequency cutoff val: {high_freq_cut}")
        
        axs[1, 0].imshow(final)
        axs[1, 0].set_title(f"Hybrid Image Result up close")
        
        axs[1, 1].imshow(final)
        axs[1, 1].set_title(f"Hybrid Image Result far away")
        plt.tight_layout()
        plt.show()
        
        freq_img1 = view_frequency_bode_graph(img1)
        freq_img2 = view_frequency_bode_graph(img2)
        
        lpf_freq_img1 = view_frequency_bode_graph(low_pass_img)
        hpf_freq_img2 = view_frequency_bode_graph(high_pass_img)
        
        hybrid_freq = view_frequency_bode_graph(final)
        
        f, axs = plt.subplots(3, 3)
        axs[0, 0].imshow(freq_img1)
        axs[0, 0].set_title(f"frequency domain of img 1")
        
        axs[0, 1].imshow(freq_img2)
        axs[0, 1].set_title(f"frequency domain of img 2")
        
        axs[1, 0].imshow(lpf_freq_img1)
        axs[1, 0].set_title(f"frequency of low pass filter on img 1")
        
        axs[1, 1].imshow(hpf_freq_img2)
        axs[1, 1].set_title(f"frequency of high pass filter on img 2")
        
        axs[0, 2].imshow(hybrid_freq)
        axs[0, 2].set_title(f"frequency of hybrid image")
        
        axs[1, 2].imshow(img1)
        axs[1, 2].set_title(f"img 1")
        
        axs[2, 0].imshow(img2)
        axs[2, 0].set_title(f"img 2")
        plt.tight_layout()
        plt.show()
    return final
def hybrid_image_with_color(img1: np.array, img2: np.array, \
    low_freq_cut: list[float] = [1.0], \
        high_freq_cut: list[float] = [1.0], display=False) -> np.array:
    assert len(img1.shape) > 2 and img1.shape[2] == img2.shape[2]
    assert len(low_freq_cut) == img1.shape[2] and len(high_freq_cut) == img1.shape[2]
    channels = []
    for channel in range(img1.shape[2]):
        channels.append(hybrid_image(img1[:, :, channel], img2[:, :, channel], \
            low_freq_cut=low_freq_cut[channel], high_freq_cut=high_freq_cut[channel]))
    
    result = np.dstack(channels)
    if display:
        f, axs = plt.subplots(2, 2)
        axs[0, 1].imshow(img1)
        axs[0, 1].set_title("img1")
        
        axs[0, 0].imshow(img2)
        axs[0, 0].set_title("img2")
        
        axs[1, 1].imshow(result)
        axs[1, 1].set_title(f"Result on colored image with low frequencies (RGB Order) \nas {low_freq_cut} \nand high frequencies as\n {high_freq_cut}")
        
        axs[1, 0].imshow(result)
        axs[1, 0].set_title("result zoomed out")
        plt.tight_layout()
        plt.show()
    return result
def blur(img, gauss_kernel=None, std: float=1.0, ksize_func = lambda std: 2 * int(3 * std) + 1) -> np.array:
    """Performs a gaussian blur on IMG and returns result as float 64 np array

    Args:
        img (np.array): The numpy matrtix repr of image to use
        gauss_kernel (np.array, None): Give a gauss kernel for gaussian blur operation. Defaults to None.
        std (float): standard deviation or cutoff frequency value. Defaults to 1.0.

    Returns:
        np.array: The low pass filter output of IMG
    """
    if gauss_kernel is None:
        g = cv2.getGaussianKernel(ksize= ksize_func(std), sigma=std)
        gauss_kernel = g @ g.T
    base: np.array = np.zeros(img.shape)
    float_img = convert_img_to_float(img)
    if len(float_img.shape) == 3 or len(float_img.shape) == 4:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            threads = []
            for i in tqdm(range(float_img.shape[2]), "Process gaussian blur on channels: "):
                threads.append(executor.submit(convolve2d, float_img[:, :, i], gauss_kernel, mode='same', boundary='symm'))
            results = [thread.result() for thread in threads]
            for i, result in enumerate(results):
                base[:, :, i] = result
    else:
        assert len(float_img.shape) == 2
        
        base = convolve2d(float_img, gauss_kernel, mode='same', boundary='symm')
    return base

def gauss_stack(img: np.array, std: float = 1.0, lvls: int = 1, \
    ksize_func=lambda std: 2 * int(3 * std) + 1) -> list[np.array]:
    """Given an IMG, create a gaussian stack with a depth of 
    LVLS   

    Args:
        img (np.array): image matrix to use
        std (float): standard deviation or sigma value for 
        gaussian kernel. Defaults to 1.0.
        lvls (int): number of levels to generate. Defaults to 1.

    Returns:
        list[np.array]: list of image matrices blurred by a repeated 
        amount for each level in LVLS
    """
    # g = cv2.getGaussianKernel(ksize=int(1 + 6 * std), sigma=std)
    # gauss_kernel = g @ g.T
    gauss_stack: list[np.array] = []
    tWalker: np.array = np.copy(img)
    if np.issubdtype(tWalker.dtype, np.integer):
        tWalker = convert_img_to_float(tWalker)
    
    for i in range(lvls):
        gauss_stack.append(convert(np.copy(tWalker)))
        tWalker = blur(tWalker, std= (2 ** (i)) * std, ksize_func=ksize_func)
    
    return gauss_stack

def laplace_stack(gauss_stack: list[np.array]) -> list[np.array]:
    
    l_stack: list[np.array] = []
    
    for i in range(len(gauss_stack) - 1):
        first_blur = gauss_stack[i]
        next_blur = gauss_stack[i + 1]
        result = cv2.subtract(first_blur, next_blur)
        # result = convert(result)
        l_stack.append(result)
    
    l_stack.append(gauss_stack[-1])
    
    return l_stack
def sharpen_channel(img, sigma=1.0):
    std = sigma 
    return img - blur(img, std=std)

def sharpen(img, sigma=1.0):
    if len(img.shape) == 2:
        return sharpen_channel(img, sigma)
    elif len(img.shape) == 3:
        base = np.zeros(img.shape)
        for i in range(img.shape[2]):
            base[:, :, i] = sharpen_channel(base[:, :, i], sigma)
        return base
    
def blend(mask_gstack: list[np.array], lstack_img1: list[np.array], lstack_img2: list[np.array]):
    """Performs multiresolution blending via gaussian and laplace stacks 
    according to formula for Laplacian Pyramid Blending.

    Args:
        mask_gstack (list[np.array]): gaussian stack of mask
        lstack_img1 (list[np.array]): laplacian stack of img1
        lstack_img2 (list[np.array]): laplacian stack of img2

    Returns:
        tuple[list[np.array]]: returns LS (Laplacian stack of combined imgs and mask),
        lefts (Laplacian stack of img1 with mask), 
        rights (Laplacian stack of img2 with mask)
    """
    zero_start = np.zeros(lstack_img1[-1].shape)
    LS = []
    lefts = []
    rights = []
    levels = len(lstack_img1)
    LA = lstack_img1
    LB = lstack_img2
    assert len(LA) == len(LB)
    GR = mask_gstack
    assert len(GR) == len(LB)
    
    for lvl in range(levels - 1, -1, -1):  
        
        left = multiply_mask_to_img(GR[lvl], LA[lvl])
        
        right = multiply_mask_to_img((1 - GR[lvl]), LB[lvl])
        if lvl == levels - 1:
            lefts.append(left)
            rights.append(right)
            LS.append(left + right)
        else:
            lefts.append(lefts[-1] + left)
            rights.append(rights[-1] + right)
            LS.append(LS[-1] + (left + right))
        
    return LS, lefts, rights
def plot_pyramids(ls, left, right, std):
    lvls = len(ls)
    f, axs = plt.subplots(lvls, 3, figsize=(17, 3 * lvls))
    for j in range(3):
        for i in range(lvls):
            if j == 0:
                title = f"image 1 at lvl: {i} with sigma: {(2 ** (i)) * std}"
                img = normalize_matrix(left[i])
            elif j == 1:
                title = f"combined img at lvl: {i} with sigma: {(2 ** (i)) * std}"
                img = normalize_matrix(ls[i])
            else:
                title = f"image 2 at lvl: {i} with sigma: {(2 ** (i)) * std}"
                img = normalize_matrix(right[i])
            
            axs[i, j].set_title(title)
            axs[i, j].imshow(img)
    plt.tight_layout()
    plt.show()
def blend_with_ui(img1_path, img2_path, std, lvls):
    img1 = skio.imread(img1_path) / 255
    img2 = skio.imread(img2_path) / 255
    img1, img2 = align_images(img1, img2)
    
    regionSelect = Selector(img1)
    mask = regionSelect.get_mask()
    
    gstack1 = gauss_stack(img1, std=std, lvls=lvls, ksize_func = lambda x: int(2 * x) + 1 - (int(2 * x) % 2))
    gstack2 = gauss_stack(img2, std=std, lvls=lvls, ksize_func = lambda x: int(2 * x) + 1 - (int(2 * x) % 2))
    
    lstack1 = laplace_stack(gstack1)
    lstack2 = laplace_stack(gstack2)
    
    gstackm = gauss_stack(mask, std=std, lvls=lvls, ksize_func = lambda x: int(2 * x) + 1 - (int(2 * x) % 2))
    
    ls, lefts, rights = blend(gstackm, lstack1, lstack2)
    
    plot_pyramids(ls, lefts, rights, std)
def main():
    # #Part 1
    finite_diff_op()
    std = 1.0
    parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    fileroot = osp.join(parent_dir,"images")
    result_root = osp.join(parent_dir, "results")
    # Part 2.1
    
    filename = osp.join(fileroot,"taj.jpg")
    unsharpen_mask_filter(filename=filename, std=std, save=False)
    filename = osp.join(fileroot, "da_bean.jpg")
    save_dest = osp.join(result_root, "killer_bean_sharp.png")
    img = unsharpen_mask_filter(filename, std, save=True, save_dest=save_dest)
    filename = osp.join(fileroot, "beany.jpg")
    save_dest = osp.join(result_root, "beany_res_sharp.png")
    unsharpen_mask_filter(filename, std, save=True, save_dest=save_dest)
    filename = osp.join(result_root, "killer_bean_sharp.png")
    save_dest = osp.join(result_root, "killer_bean_resharpened.png")
    sharp_image = get_image_as_float(filename)
    blurred = blur(sharp_image)
    resharpened = unsharpen_mask_filter(blurred, std, save=True, save_dest=save_dest)
    f, axs = plt.subplots(1, 3)
    axs[0].imshow(sharp_image)
    axs[0].set_title("Original Sharpened Image")
    axs[1].imshow(resharpened)
    axs[1].set_title("Blurred and Resharpened Image")
    axs[2].imshow(blurred)
    axs[2].set_title("Low Pass Filter of sharpened image")
    plt.tight_layout()
    plt.show()
    
    # part 2.2 Hybrid images
    
    # bean images
    img1_path = osp.join(fileroot, "killer_bean.jpg")
    img2_path = osp.join(fileroot, "jet_bean.png")
    result_path = osp.join(result_root, "killer_jet.jpg")
    
    # shiba and boy
    img1_path = osp.join(fileroot, "shiba.png")
    img2_path = osp.join(fileroot, "him.jpeg")
    result_path = osp.join(result_root, "shiba_boy.jpg")
    
    #shiba and sideeye
    img1_path = osp.join(fileroot, "shiba_dumb.jpg")
    img2_path = osp.join(fileroot, "side_eye.png")
    result_path = osp.join(result_root, "shiba_side_eye_color.jpg")
    img1 = skio.imread(img1_path) / 255
    img2 = skio.imread(img2_path) / 255
    
    img1, img2 = align_images(img1, img2)
    
    img1_grey = sk.color.rgb2gray(img1)
    img2_grey = sk.color.rgb2gray(img2)
    #higher sigma means more to see in high freq --> higher sigma means lower low pass
    result = hybrid_image_with_color(img1, img2, low_freq_cut=[5.5] * 3, high_freq_cut=[4.0] * 3, display=True)
    plt.imsave(result_path, result)
    result_path = osp.join(result_root, "shiba_side_eye.jpg")
    result = hybrid_image(img1_grey, img2_grey, low_freq_cut=5.5, high_freq_cut=4.0, display=True)
    plt.imsave(result_path, result)
    
    
    # old vs young
    img_path = osp.join(fileroot, "old_vs_young.JPG")
    result_path = osp.join(result_root, "old_vs_young.jpg")
    img = skio.imread(img_path) / 255
    half = img.shape[0] // 2
    img2 = img[:half, :, :]
    img1 = img[half:, :, :]
    
    img1, img2 = align_images(img1, img2)
    
    img1 = sk.color.rgb2gray(img1)
    img2 = sk.color.rgb2gray(img2)
    
    result = hybrid_image(img1, img2, low_freq_cut=2.0, high_freq_cut=2.2, display=True)
    
    #part 2.3
    std = 1.0
    filename = osp.join(fileroot, "orple.jpg")
    oraple = skio.imread(filename)
    
    gstack = gauss_stack(oraple, std=std, lvls=5, ksize_func = lambda x: int(2 * x) + 1 - (int(2 * x) % 2))
    gstack = [convert(img) for img in gstack]
    lstack = laplace_stack(gstack)
    f, axs = plt.subplots(len(gstack), 2, figsize=(10, 10))
    
    recreate = np.copy(gstack[-1])
    for i in range(len(lstack) - 1, -1, -1):
        recreate += lstack[i]
    #gstack place
    for i in range(len(gstack)):
        img = gstack[i]
        axs[i, 0].set_title(f"Gaussian Stack at level: {i} and \n gaussian blur w/ sigma: {(2 ** (i)) * std}")
        axs[i, 0].imshow(img)
        
    #lstack place
    for i in range(len(gstack)):
        if i + 1 == len(gstack):
            axs[i, 1].set_title(f"Laplace Stack Combined with lowest freq of Gaussian Stack")
            axs[i, 1].imshow(normalize_matrix(recreate))
        else:
            axs[i, 1].set_title(f"Laplace Stack at level: {i} and \n gaussian blur w/ sigma: {(2 ** (i)) * std}")
            axs[i, 1].imshow(normalize_matrix(lstack[i]))
        
    
    plt.tight_layout()
    plt.show()
    
    f, axs = plt.subplots(2, 1)
    
    recreate = normalize_matrix(recreate)
    axs[0].imshow(recreate)
    axs[0].set_title("Combined laplacian layers")
    axs[1].imshow(oraple)
    axs[1].set_title("Original image")
    plt.tight_layout()
    plt.show()
    
    # part 2.4
    
    apple = skio.imread(osp.join(fileroot, "apple.jpeg")) / 255
    orange = skio.imread(osp.join(fileroot, "orange.jpeg")) / 255
    
    lvls = 5
    apple_gstack = gauss_stack(apple, std=std, lvls=lvls, ksize_func = lambda x: int(2 * x) + 1 - (int(2 * x) % 2))
    orange_gstack = gauss_stack(orange, std=std, lvls=lvls, ksize_func = lambda x: int(2 * x) + 1 - (int(2 * x) % 2))
    
    apple_lstack = laplace_stack(apple_gstack)
    orange_lstack = laplace_stack(orange_gstack)
    
    mask = np.zeros(apple.shape[:2])
    mask[:, :mask.shape[1] // 2 - 5] = np.ones(mask[:, :mask.shape[1] // 2 - 5].shape)
    # mask = feather_mask(apple, 2, 0.0)
    
    mask_gstack = gauss_stack(mask, std=std, lvls=lvls, ksize_func = lambda x: int(2 * x) + 1 - (int(2 * x) % 2))
   
    LS, lefts, rights = blend(mask_gstack, apple_lstack, orange_lstack)
    
    plot_pyramids(LS, lefts, rights, std)
    
    #part 2.4 part 2 custom
    lvls = 5
    shiba_path = osp.join(fileroot, "shiba.png")
    shiba_d_path = osp.join(fileroot, "shiba_dumb.jpg")
    
    blend_with_ui(shiba_path, shiba_d_path, std, lvls)
    
    him_path = osp.join(fileroot, "him.jpeg")
    blend_with_ui(shiba_path, him_path, std, lvls)
    
    s = "stop"
if __name__ == '__main__':
    main()
    