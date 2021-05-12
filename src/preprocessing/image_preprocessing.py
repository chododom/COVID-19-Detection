"""
This file contains definitions of preprocessing functions and their helper functions used during our experimentation with image preprocessing methods.

Author: Dominik Chodounský
Institution: Faculty of Information Technology, Czech Technical University in Prague
Last edit: 2021-02-16
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_preprocessing(imgs, preprocessor, figsize, mode=None):
    """
    Displays a vizualization of the individual steps of the preprocessing function. The use case for this functionality is purely for demonstrative purposes.
    
    Parameters
    ----------
    imgs : NumPy.ndarray
        List of images that contain the result of individual steps of the vizualized algorithm.
    preprocessor : fun
        Preprocessing function to be demonstrated.
    figsize : (width, height)
        Tuple determining the size of the displayed figure.
    mode : str
        If a string is provided, it will be either 'poly' or 'hull' to signify the type of contour approximation that will take place in diaphragm removal preprocessing technique.
        All other cases use a None mode. Default value: None.
    """
    
    cnt = len(imgs)
    plt.figure(figsize=figsize)
    iter = 1
    for i in range(cnt):
        if mode:
            stages = list(preprocessor(imgs[i], mode=mode))
        else:
            stages = list(preprocessor(imgs[i]))
        columns = len(stages)
        for j in range(columns):
            plt.subplot(cnt, columns, iter)
            plt.imshow(np.squeeze(stages[j]), vmin=0, vmax=255, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            iter += 1
    plt.show()


def remove_diaphragm(img, img_size=224, cutoff=0.8, mode='poly'):
    """
    Method of removing the high-intensity diaphragm region. Thresholding is performed on a grayscale version of the image to convert it into a binary mask. The contour of the biggest remaining object
    is found, selected, smoothened by morphological operators and the area is removed from the original image. This version is then copied where one copy is passed through a bilateral filter,
    and another has its contrast increased with histogram equalization. These two copies along with the original image with removed diaphragm are combined into a pseudo-colour image.
    
    Parameters
    ----------
    img : NumPy.ndarray
        Image to be preprocessed.
    img_size : int
        Width and height that the image will be resized to. Default value: 224.
    cutoff : float
        Cut-off value used in the thresholding. Default value: 0.8.
    mode : str
        Defines the method of selecting the contour. Option 'poly' approximates it with a polygon, option 'hull' encloses the contour with a convex hull. Default value: poly.
        
    Returns
    -------
    pipeline : list
        List of 7 images that were gathered throughout the various preprocessing steps.
    """
    
    # reduce dimensionality to single-channel grayscale image and resize to target width and height
    if len(np.squeeze(img).shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img.astype('uint8'), (img_size, img_size))

    max = np.amax(img, axis=None)
    min = np.amin(img, axis=None)
    
    # calculate threshold to perform segmentation with
    threshold = min + cutoff * (max - min)
    _, bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # find remaining contours
    drawn, cnt, conts = find_contours(bin, fill=True)
    
    # if no contours were found, return what is possible
    if cnt == 0:
        equalized = cv2.equalizeHist(img)
        blurred = cv2.bilateralFilter(img, 9, 75, 75)
        return bin, img, img, img, equalized, blurred, cv2.merge((img, equalized, blurred))

    biggest = biggest_contour(conts)
    
    # enclose biggest contour with a convex hull and smoothen it with morphological operators
    if mode == 'hull':
        hull = cv2.convexHull(biggest, returnPoints = True)

        approx_bin = draw_contours(img, [hull], fill=True)
        approx_bin = smoothen(approx_bin)

        marked = cv2.drawContours(img.copy(), [hull], -1, (0, 0, 0), thickness=2)

    # approximate biggest contour with a polygon and smoothen it with morphological operators
    elif mode == 'poly':
        perimeter = cv2.arcLength(biggest, True)
        epsilon = 0.01 * cv2.arcLength(biggest, True)
        poly = cv2.approxPolyDP(biggest, epsilon, True)

        approx_bin = draw_contours(img, [poly], fill=True)
        approx_bin = smoothen(approx_bin)

        marked = cv2.drawContours(img.copy(), [poly], -1, (0, 0, 0), thickness=2)

    # create version of image with removed diaphragm, equalized histogram, bilateral filtering, and combine them into pseudo-colour image
    removed_diaphragm = cv2.bitwise_and(img.copy(), img.copy(), mask=negative(approx_bin))
    equalized = cv2.equalizeHist(removed_diaphragm)
    blurred = cv2.bilateralFilter(removed_diaphragm, 9, 75, 75)
    return img, drawn, marked, removed_diaphragm, equalized, blurred, cv2.merge((removed_diaphragm, equalized, blurred))
    

def rgb_histogram_equalization(img, img_size=224):
    """
    Method of performing histogram equalization on RGB images. The equalization is performed on the luminance channel in YCrCb colour space.
    Colour space conversion inspired by source: #https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
    
    Parameters
    ----------
    img : NumPy.ndarray
        Image to be preprocessed.
    img_size : int
        Width and height that the image will be resized to. Default value: 224.

    Returns
    -------
    pipeline : list
        List of 2 images that were gathered throughout the various preprocessing steps.
    """
    
    # denoise image with bilateral filter and extract luminance component from YCrCb colour space
    blurred = cv2.bilateralFilter(img.astype('uint8'), 9, 75, 75)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)

    # perform histogram equalization and convert back to RGB
    channels[0] = cv2.equalizeHist(channels[0])
    ycrcb = cv2.merge(channels)
    new_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    new_img = cv2.resize(new_img.astype('uint8'), (img_size, img_size))
    return cv2.resize(img.astype('uint8'), (img_size, img_size)), new_img


def rgb_clahe(img, img_size=224, clip_limit=3):
    """
    Method of performing contrast limited adaptive histogram equalization (CLAHE) on RGB images. The equalization is performed on the luminance channel in YCrCb colour space.
    Colour space conversion inspired by source: #https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
    
    Parameters
    ----------
    img : NumPy.ndarray
        Image to be preprocessed.
    img_size : int
        Width and height that the image will be resized to. Default value: 224.
    clip_limit : int
        Maximum number of pixels with a common intensity value after equalization.

    Returns
    -------
    pipeline : list
        List of 2 images that were gathered throughout the various preprocessing steps.
    """
    
    # denoise image with bilateral filter and extract luminance component from YCrCb colour space
    blurred = cv2.bilateralFilter(img.astype('uint8'), 9, 75, 75)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)

    # perform CLAHE and convert back to RGB
    clahe = cv2.createCLAHE(clip_limit) 
    channels[0] = clahe.apply(channels[0])

    ycrcb = cv2.merge(channels)
    new_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    new_img = cv2.resize(new_img.astype('uint8'), (img_size, img_size))
    return cv2.resize(img.astype('uint8'), (img_size, img_size)), new_img
    

def find_contours(img_bin, min_area=0, max_area=np.inf, fill=False, external=False):
    """
    Finds contours of objects within an image. Function inspired by a similar adaptation used in course BI-SVZ at FIT CTU in Prague.
    
    Parameters
    ----------
    img_bin : NumPy.ndarray
        Binary image to find contours in.
    min_area : int
        Minimum area of a contour to be considered in the selection. Default value: 0.
    max_area : int
        Maximum area of a contour to be considered in the selection. Default value: np.inf.
    fill : bool
        Whether to fill in the found contours. Default value: False.
    external : bool
        Whether to include only the external border of the object. Default value: False.
    
    Returns
    -------
    pipeline : list
        List of 3 images that were gathered throughout the various preprocessing steps.        
    """
    
    if external:
        mode = cv2.RETR_EXTERNAL
    else:
        mode = cv2.RETR_LIST

    conts, hierarchy = cv2.findContours(img_bin, mode, cv2.CHAIN_APPROX_SIMPLE)
    
    conts = [c for c in conts if cv2.contourArea(c) > min_area and cv2.contourArea(c) < max_area]

    if fill:
        thick = thick = cv2.FILLED
    else:
        thick = 2

    drawn = cv2.drawContours(np.zeros(img_bin.shape, dtype=np.uint8), conts, -1, color=(255, 0, 0), thickness=thick)
    return drawn, len(conts), conts


def draw_contours(img, conts, fill=False):
    """
    Draws contours on top of an image. Function inspired by a similar adaptation used in course BI-SVZ at FIT CTU in Pra
    
    Parameters
    ----------
    img : NumPy.ndarray
        Image to draw contours on.
    conts : list
        List of contours.
    fill : bool
        Whether to fill in the drawn contours. Default value: False.
    
    Returns
    -------
    drawn : NumPy.ndarray
        Image with given contours drawn on top of it.     
    """
    
    if fill:
        thick = thick = cv2.FILLED
    else:
        thick = 2
    drawn = cv2.drawContours(np.zeros(img.shape, dtype=np.uint8), conts, -1, color=(255, 0, 0), thickness=thick)
    return drawn


def biggest_contour(conts):
    """
    Finds contour with the biggest area in a list of given contours.
    
    Parameters
    ----------
    conts : list
        List of contours.
        
    Returns
    -------
    cont : NumPy.ndarray
        Contour with the largest area. 
    """
    
    areas = [cv2.contourArea(c) for c in conts]
    max_index = np.argmax(areas)
    return conts[max_index]


def smoothen(img_bin, kernel_size=15, iterations=50):
    """
    Smoothens a binary image with a set of morphological operators. First, opening is applied, then dilation and finally erosion.
    
    Parameters
    ----------
    img_bin : NumPy.ndarray
        Binary image to smoothen.
    kernel_size : int
        Size of the kernels used by the morphological operators (must be odd number). Default value: 15.
    iterations : int
        Number of iterations of each of the steps in the smoothening process. Default value: 50.
        
    Returns
    -------
    erosion : NumPy.ndarray
        Smoothened version of the input image.
    """
    
    # closing
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    filled = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, struct)

    # dilation
    dilation = cv2.dilate(filled, np.ones((kernel_size, kernel_size), np.uint8), iterations)

    # erosion
    erosion = cv2.erode(dilation, np.ones((kernel_size, kernel_size), np.uint8), iterations)

    return erosion


def negative(img):
    """
    Creates a negative version of a given image.
    
    Parameters
    ----------
    img_bin : NumPy.ndarray
        Binary image to create the negative of.
        
    Returns
    -------
    neg : NumPy.ndarray
        Negative version of the input image.
    """
    
    neg = cv2.bitwise_not(img)
    return neg