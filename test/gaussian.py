import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to a color image."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)  # Add noise to the image
    return noisy_image

# Load the color image
image = cv2.imread("encrypted.png")

# Apply Gaussian noise
noisy_image = add_gaussian_noise(image)

# Save the noisy image
cv2.imwrite("noisy_image.png", noisy_image)

# Display images
#cv2.imshow("Original Image", image)
cv2.imshow("Noisy Image (Gaussian Attack)", noisy_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
