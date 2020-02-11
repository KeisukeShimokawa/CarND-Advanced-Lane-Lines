import cv2
import json
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_all_images(images_path):
    """
    plot all passed image by 2 columns
    
    args:
        images_path(list of str): list of image path you want to plot
    """
    
    nimage = len(images_path)
    ncols = 2
    nrows = nimage//ncols if (nimage%ncols==0) else (nimage//ncols) + 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))
    axes = axes.flatten()
    for index, path in enumerate(images_path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[index].imshow(img)
        axes[index].set_title(path, fontsize=30)
        axes[index].set_xticks([])
        axes[index].set_yticks([])
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
    fig.tight_layout()
    
    
def undistort_and_save_image(images_path, data):
    """
    undistort image based on camera matrix and distortion coefficient.
    save image to the same directory of original image located.
    
    args:
        images_path(list of str): list of image path you want to undistort and save.
        data(dict): dictionary object store camera matrix and distortion coefficients.
    """
    
    img = cv2.imread(path)
    dst = cv2.undistort(img, data['mtx'], data['dist'], None, data['newcameramtx'])
    
    dst_path_name = path.replace('.jpg', '_dst.jpg')
    cv2.imwrite(dst_path_name, dst)
    
    
def sobel_filter(sobel):
    """
    apply normalization for image applying sobel filter
    
    args:
        sobel(ndarray): 2-dim image array
    """
    
    abs_sobel = np.absolute(sobel)
    normalized_sobel= np.uint8(255 * abs_sobel / np.max(abs_sobel))
    return normalized_sobel


def apply_threshholding(gray, thresh_min=0, thresh_max=255):
    """
    create binary image based on two threshhold value toward gray scale image.
    
    args:
        gray(ndarray): 2-dim ndarray
        thresh_min(int): threshhold value for minimum activation
        thresh_max(int): threshhold value for maximum activation
    """
    
    binary = np.zeros_like(gray)
    binary[(gray >= thresh_min) & (gray <= thresh_max)] = 1
    return binary


def get_hsv_color_range(hsv, lower, upper):
    """
    get selected image between lower and upper hsv color range
    
    args:
        hsv: 3-dim OpenCV format image of HSV
        lower: lowest value of hsv
        upper: highest value of hsv
    """
    
    if (type(lower) == list) or (type(upper) == list):
        lower = np.array()
        upper = np.array()
    
    frame_mask = cv2.inRange(sample_img_hsv, lower, upper)
    dst = cv2.bitwise_and(sample_img_hsv, sample_img_hsv, mask=frame_mask)
    return dst