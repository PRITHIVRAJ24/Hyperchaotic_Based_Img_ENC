import cv2
import numpy as np

def mse(image1, image2):
    # Compute the Mean Squared Error (MSE) for each channel
    err = np.mean((image1 - image2) ** 2, axis=(0, 1))
    return err

def psnr(image1, image2):
    mse_values = mse(image1, image2)
    if np.any(mse_values == 0):
        return float('inf')
    max_pixel = 255.0
    psnr_values = 20 * np.log10(max_pixel / np.sqrt(mse_values))
    return psnr_values

# Load images (original and encrypted)
original = cv2.imread("application\src\Sample\Lenna.png")
encrypted = cv2.imread("application\src\output\encrypted_test.png")

# Ensure images are loaded
if original is None or encrypted is None:
    print("Error: One or both images could not be loaded.")
else:
    mse_values = mse(original, encrypted)
    psnr_values = psnr(original, encrypted)
    
    print(f"MSE: {mse_values}")
    print(f"PSNR: {psnr_values} dB")