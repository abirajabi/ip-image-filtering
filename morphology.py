'''
Morphology :
    1. Erosion
    2. Dilation
    3. Closing
    4. Opening
    5. Morphological Gradient
    6. Top hat
    7. Black Hat
'''
#  in here kernel means sturcturing element
#  all kernel use the cross structuring element / +

import cv2
import numpy as np

def erosion(image, kernel_size, n_iteration):
    i = n_iteration
    ks = kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(ks, ks))
    erosion = cv2.erode(image, kernel, iterations=i)

    cv2.imshow("Erosion", erosion)

def dilation(image, kernel_size, n_iteration):
    i = n_iteration
    ks = kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(ks, ks))
    dilation = cv2.dilate(image, kernel, iterations=i)

    cv2.imshow("Dilation", dilation)

def morphological_transformation(image, morph_type, kernel_size):
    ks = kernel_size
    struct_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (ks, ks))
    
    def make_morphology(_type):
        switcher = {
            cv2.MORPH_OPEN: cv2.morphologyEx(image, cv2.MORPH_OPEN, struct_element),
            cv2.MORPH_CLOSE: cv2.morphologyEx(image, cv2.MORPH_CLOSE, struct_element),
            cv2.MORPH_GRADIENT: cv2.morphologyEx(image, cv2.MORPH_GRADIENT, struct_element),
            cv2.MORPH_TOPHAT: cv2.morphologyEx(image, cv2.MORPH_TOPHAT, struct_element),
            cv2.MORPH_BLACKHAT: cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, struct_element)
        }
        return switcher.get(_type, "Nothing")
    
    filtered_image = None
    filtered_image = make_morphology(morph_type)

    cv2.imshow("Transformed", filtered_image)
