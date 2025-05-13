import cv2
import numpy as np



def mse(imageA, imageB):
    """Compute the Mean Squared Error (MSE) between two images."""
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')  # PSNR is infinite if images are identical
    return 10 * np.log10((255 ** 2) / mse_value)
image =cv2.imread("application/src/Sample/Lenna.png")
# Load the color image
noisy_image = cv2.imread("decrypted.png")


# Apply Gaussian filter (5x5 kernel)
filtered_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)

# Save noisy and filtered images
#cv2.imwrite("noisy_image.jpg", noisy_image)
cv2.imwrite("filtered_image.png", filtered_image)

# Compute MSE and PSNR
mse_value = mse(image, filtered_image)
psnr_value = psnr(image, filtered_image)

# Print MSE and PSNR values
print(f"MSE: {mse_value}")
print(f"PSNR: {psnr_value} dB")

# Display images
#cv2.imshow("Original Image", image)
#.imshow("Noisy Image (Gaussian Attack)", noisy_image)
cv2.imshow("Filtered Image (Gaussian Filter)", filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
