import cv2
import numpy as np
import random

def add_salt_and_pepper_noise_color(image, noise_ratio=1):
    
    noisy_image = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    num_noise_pixels = int(noise_ratio * total_pixels)


    for _ in range(num_noise_pixels // 2):
        x, y = random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1)
        noisy_image[x, y] = [255, 255, 255] 


    for _ in range(num_noise_pixels // 2):
        x, y = random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1)
        noisy_image[x, y] = [0, 0, 0]  

    return noisy_image

def mse(imageA, imageB):
    
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):

    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')  
    return 10 * np.log10((255 ** 2) / mse_value)

encrypted_image = cv2.imread("application\src\output\encrypted.png")

noisy_encrypted_image = add_salt_and_pepper_noise_color(encrypted_image, noise_ratio=0.3)

cv2.imwrite("application/src/output/noise_encrytped.png", noisy_encrypted_image)



cv2.imshow("Noisy Encrypted Image", noisy_encrypted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
