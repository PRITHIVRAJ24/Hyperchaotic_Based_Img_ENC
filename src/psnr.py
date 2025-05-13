import cv2
import numpy as np

def calculate_psnr(original, decrypted):
    # Convert images to float32 for precise calculations
    original = original.astype(np.float32)
    decrypted = decrypted.astype(np.float32)
    
    # Compute Mean Squared Error (MSE)
    mse = np.mean((original - decrypted) ** 2)
    
    if mse == 0:
        return float('inf')  # Infinite PSNR (no difference between images)

    # Compute PSNR
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

# Load original and decrypted images
original_img = cv2.imread("D:/Final Project/application/src/Sample/Lenna.png")
decrypted_img = cv2.imread("D:/Final Project/application/src/encrypted_test.png")

# Ensure images have the same dimensions
if original_img.shape != decrypted_img.shape:
    print("Error: Image dimensions do not match!")
else:
    psnr_value = calculate_psnr(original_img, decrypted_img)
    print(f"PSNR Value: {psnr_value:.2f} dB")
