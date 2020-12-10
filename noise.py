'''
    # 1. salt and pepper
    # 2. impulse noise; a. exponential, b. gaussian
    # 3. speckle
'''

import numpy as np
import cv2

# not done yet
def gaussian_noise(image, mean, var):
    row,col,ch= image.shape
    std = var**0.5
    gauss = np.random.normal(mean, std,(row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + gauss

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)

    cv2.imshow("noisy", noisy_image)

def salt_pepper_noise(image, salt_ratio, amount):
    img = image
    row, col, ch = img.shape
    out = np.copy(img)

    # Salt
    num_salt = np.ceil(amount * img.size * salt_ratio)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    out[coords] = 1

    # Pepper
    num_pepper = np.ceil(amount * image.size * (1. - salt_ratio))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 1

    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    out = out.astype(np.uint8)

    cv2.imshow("Salt and pepper", out)
    

def speckle_noise(image):
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    noisy_image = image + image * gauss

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)

    cv2.imshow("Speckle noise", noisy_image)

def poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_image = np.random.poisson((image * vals)/float(vals))

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)

    cv2.imshow("Poisson noise", noisy_image)