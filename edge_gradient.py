'''
    Image gradient: Laplacian, Scharr, Sobel
    Edge detection: Canny, Robert, Prewitt,  
'''
import cv2
import numpy as np

def laplacian(image):
    laplacian_g = cv2.Laplacian(image, cv2.CV_64F)
    # abs_laplacian64f = np.absolute(laplacian_g)
    # laplacian_g = np.uint8(abs_laplacian64f)
    
    cv2.imshow("Laplacian gradient", laplacian_g)

def sobel(image, kernel_size):
    # sobel = cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=kernel_size)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    abs_sobelx64f = np.absolute(sobelx)
    sobelx_8u = np.uint8(abs_sobelx64f)

    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    abs_sobely64f = np.absolute(sobely)
    sobely_8u = np.uint8(abs_sobely64f)
    
    sobel = sobelx_8u + sobely_8u
    cv2.imshow("Sobel", sobel)
    # cv2.imshow("Sobel X", sobelx_8u)
    # cv2.imshow("Sobel Y", sobely_8u)

def canny(image, treshold1, treshold2):
    canny = cv2.Canny(image, treshold1, treshold2)
    cv2.imshow("Canny Edge Detection", canny)

def prewitt(image):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(image, -1, kernelx)
    img_prewitty = cv2.filter2D(image, -1, kernely)
    cv2.imshow("Prewitt", img_prewittx + img_prewitty)

def robert(image):
    kernelx = np.array([[1,0],[0,-1]])
    kernely = np.array([[0,-1],[1,0]])
    img_robertx = cv2.filter2D(image, -1, kernelx)
    img_roberty = cv2.filter2D(image, -1, kernely)
    cv2.imshow("Robert", img_robertx + img_roberty)