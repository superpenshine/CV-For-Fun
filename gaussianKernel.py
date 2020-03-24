import numpy as np
import cv2
from scipy.ndimage.filters import convolve
import scipy
import math

def minMaxNorm(img):
    minv = np.amin(img)
    value_range = np.amax(img) - minv
    img = (img - minv) / value_range

    return img

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / math.sqrt(2.0 * np.pi * sigma**2)
    # normal = 1
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

    return g

def dog(img, k1, k2):
    blur1 = convolve(img, gaussian_kernel(25, k1))
    blur2 = convolve(img, gaussian_kernel(25, k2))

    return minMaxNorm(blur2 - blur1)

def dog2(img, k1, k2):
    blur1 = scipy.ndimage.gaussian_filter(img, k1)
    blur2 = scipy.ndimage.gaussian_filter(img, k2)

    return minMaxNorm(blur2 - blur1)

def edge(img):
    kernel_x = np.array([[1, 0, -1]])
    kernel_y = np.array([[1], [0], [-1]])
    Ix = convolve(img, kernel_x)
    Iy = convolve(img, kernel_y)

    return np.hypot(Ix, Iy)

def edge2(img):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = convolve(img, kernel_x)
    Iy = convolve(img, kernel_y)

    return np.hypot(Ix, Iy)

img = cv2.imread('input.png',1)/255
img_gray = np.mean(np.array(img), axis=2)
img_smoothed1 = convolve(img_gray, gaussian_kernel(5, sigma=1))
img_smoothed3 = scipy.ndimage.gaussian_filter(img_gray, 2)

img_dog = dog(img_gray, 0.3, 0.4)
img_dog2 = dog2(img_gray, 1, 2)

img_edge = edge(img_smoothed3)
img_edge2 = edge2(img_smoothed3)

numpy_horizontal_concat = np.concatenate((img_smoothed1, img_dog, img_smoothed3, img_dog2, img_edge, img_edge2), axis=1)
cv2.imshow('Gaussian blur', numpy_horizontal_concat)
cv2.waitKey()
cv2.destroyAllWindows()
exit(0)
