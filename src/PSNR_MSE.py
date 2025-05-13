import cv2
import numpy as np

def calculate_psnr_mse(original_image_path, encrypted_image_path):
    original = cv2.imread(original_image_path)
    encrypted = cv2.imread(encrypted_image_path)

    if original is None or encrypted is None:
        print("Error: Could not load images. Check file paths!")
        return

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    encrypted = cv2.cvtColor(encrypted, cv2.COLOR_BGR2RGB)

    mse_values = {}
    psnr_values = {}

    for i, color in enumerate(['Red', 'Green', 'Blue']):
        mse = np.mean((original[:, :, i] - encrypted[:, :, i]) ** 2)
        mse_values[color] = mse

        if mse == 0:
            psnr_values[color] = float('inf')
        else:
            psnr_values[color] = 10 * np.log10((255 ** 2) / mse)

    return mse_values, psnr_values

original_image_path = "application/src/Sample/Lenna.png"
encrypted_image_path = "application/src/output/encrypted_test.png"

mse, psnr = calculate_psnr_mse(original_image_path, encrypted_image_path)

print("MSE Values:", mse)
print("PSNR Values:", psnr)
