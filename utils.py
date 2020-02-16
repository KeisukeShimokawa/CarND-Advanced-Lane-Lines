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
        lower = np.array(lower)
        upper = np.array(upper)
    
    frame_mask = cv2.inRange(hsv, lower, upper)
    dst = cv2.bitwise_and(hsv, hsv, mask=frame_mask)
    return dst


def gradient_magnitude(sobelx, sobely):
    """
    get the magnitude of gradients of sobel filter for both x and y direction
    
    args:
        sobelx(ndarray): normalized 2-dim image after applying x direction sobel filter 
        sobely(ndarray): normalized 2-dim image after applying y direction sobel filter
    """
    
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    return gradmag


def get_binary_mask_from_image(img, cfg):
    
    # convert BGR to Gray scale for gradient calculation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # sobel filter
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = gradient_magnitude(sobelx, sobely)
    
    sobelx_norm = sobel_filter(sobelx)
    sobely_norm = sobel_filter(sobely)
    
    sx_binary = apply_threshholding(sobelx_norm, cfg['sobelx']['min'], cfg['sobelx']['max'])
    sy_binary = apply_threshholding(sobely_norm, cfg['sobely']['min'], cfg['sobely']['max'])
    smag_binary = apply_threshholding(sobel_mag, cfg['sobelmag']['min'], cfg['sobelmag']['max'])
    
    # COLOR(HLS) Threshholding
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hls_sch_binary = apply_threshholding(img_hls[:, :, 2], cfg['hls']['min'], cfg['hls']['max'])
    
    # COLOR(HSV) Threshholding
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_binary = get_hsv_color_range(img_hsv, 
                                     cfg['hsv']['color_range_min'], 
                                     cfg['hsv']['color_range_max'])
    hsv_range_binary = apply_threshholding(hsv_binary[:, :, 2], cfg['hsv']['min'], cfg['hsv']['max'])
    
    # get binary mask
    binary = np.zeros_like(hsv_range_binary)
    binary[((sx_binary == 1) & (sy_binary == 1) & (smag_binary == 1)) | 
           (hls_sch_binary == 1) | 
           (hsv_range_binary == 1)] = 1
    
    return binary


def get_leftx_rightx_base(binary):
    
    # get bottom_half position
    bottom_half = binary[binary.shape[0] // 2:, :]
    
    # get histogram along y-axis
    histogram = np.sum(bottom_half, axis=0)
    
    # get x-axis position of both lane line
    midpoint = int(binary.shape[1] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    return leftx_base, rightx_base


def get_left_right_lane_xy(binary, leftx_base, rightx_base):
    
    height, width = binary.shape
    
    nwindows = 9
    margin = 100
    minpix = 50
    window_height = np.int(binary.shape[0] // nwindows)
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    nonzero = binary.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
    
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        
        win_y_low  = height - (window + 1) * window_height
        win_y_high = height - window       * window_height
        
        win_xleft_low   = leftx_current  - margin
        win_xleft_high  = leftx_current  + margin
        win_xright_low  = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # get y-axis index of lane line pixel
        good_left_inds = ((nonzeroy >= win_y_low) & 
                          (nonzeroy <= win_y_high) &
                          (nonzerox >= win_xleft_low) & 
                          (nonzerox <= win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & 
                           (nonzeroy <= win_y_high) &
                           (nonzerox >= win_xright_low) & 
                           (nonzerox <= win_xright_high)).nonzero()[0]
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    return leftx, lefty, rightx, righty


def get_fitting_lane_line_position(leftx, lefty, rightx, righty):
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, height-1, height)

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty


def get_lane_pts_for_fillPoly(left_fitx, right_fitx, ploty):
    left_lane_pts  = np.array([left_fitx[::-1], ploty[::-1]]).T
    right_lane_pts = np.array([right_fitx, ploty]).T
    
    lane_pts = np.vstack((left_lane_pts, right_lane_pts))
    
    return lane_pts


def get_lane_rectangle_image(binary, lane_pts):
        
    out_binary = np.zeros_like(binary)
    out_img = np.dstack((out_binary, out_binary, out_binary)) * 255
    
    cv2.fillPoly(out_img, np.int_([lane_pts]), (0, 255, 0))
    
    return out_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):

    return cv2.addWeighted(initial_img, α, img, β, γ)


def lane_finding(path, cfg, perspective_mtx):
    
    img = cv2.imread(path)
    im_size = img.shape[::-1][1:]
    
    binary = get_binary_mask_from_image(img, cfg)
    
    M = perspective_mtx['M']
    MInv = perspective_mtx['MInv']
    binary_warped = cv2.warpPerspective(binary, M, binary.shape[::-1])
    
    leftx_base, rightx_base = get_leftx_rightx_base(binary_warped)
    
    leftx, lefty, rightx, righty = get_left_right_lane_xy(binary_warped, 
                                                          leftx_base, 
                                                          rightx_base)
    
    left_fitx, right_fitx, ploty = get_fitting_lane_line_position(leftx, lefty, 
                                                                  rightx, righty)
    
    lane_pts = get_lane_pts_for_fillPoly(left_fitx, right_fitx, ploty)
    
    img_lane_warped = get_lane_rectangle_image(binary_warped, lane_pts)
    
    img_lane = cv2.warpPerspective(img_lane_warped, MInv, im_size)
    
    out_img = weighted_img(img_lane, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    return out_img