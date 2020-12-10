'''Filtering :

1. linear
2. non linear filter: median, snn
3. smoothing (blurring)
4. sharpening : a. laplacian
5. low-pass filters : a. mean, b. gaussian
6. conservative: minmax filtering

Not done yet
'''

import sys
import cv2
import numpy as np

#  convolutional 2d
def convolutional(image_container, kernel_size, depth):
    # params image, kernel size, depth
    print(type(kernel_size), type(depth))
    ks = kernel_size
    d = depth
    img = cv2.cvtColor(image_container, cv2.COLOR_BGR2RGB)

    kernel = np.ones((ks, ks), np.float32)/(ks*ks)
    blur = cv2.filter2D(img, d, kernel)

    cv2.imshow("Concolutional 2D", blur)

# averaging / mean filtering
def averaging(image_container, kernel_size):
    img = cv2.cvtColor(image_container, cv2.COLOR_BGR2RGB)
    # params image, kernel size
    ks = kernel_size
    blur = cv2.blur(img, (ks, ks))
    
    cv2.imshow("Mean filtering", blur)

# gaussian filtering 
def gaussian(image_container, kernel_size, sigmaX):
    img = cv2.cvtColor(image_container, cv2.COLOR_BGR2RGB)
    # params image, kernel size, sigmaX
    ks = kernel_size
    sx = sigmaX
    blur = cv2.GaussianBlur(img, (ks, ks), sx)

    cv2.imshow("Gaussian filtering", blur)

# median filtering
def median(image_container, kernel_size):
    img = cv2.cvtColor(image_container, cv2.COLOR_BGR2RGB)
    # params image, kernel size
    ks = kernel_size
    blur = cv2.medianBlur(img, ks)

    cv2.imshow("Median filtering", blur)

# bilateral filtering
def bilateral(image_container, d, sigma_color, sigma_space):
    img = cv2.cvtColor(image_container, cv2.COLOR_BGR2RGB)
    #  params image, d, sigmaColor, sigmaSpace
    df = d
    sc = sigma_color
    ss = sigma_space
    blur = cv2.bilateralFilter(img, df, sc, ss)

    cv2.imshow("Bilateral filtering", blur)
    