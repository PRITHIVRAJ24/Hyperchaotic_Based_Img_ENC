import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# Load color image
image = cv2.imread("application\src\output\decrypted_test.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split into R, G, B channels
r, g, b = cv2.split(image)

# Function to calculate noise percentage
def noise_percentage(original, noisy):
    diff = np.abs(original.astype(np.int16) - noisy.astype(np.int16))
    noise_pixels = np.count_nonzero(diff)
    total_pixels = original.size
    return (noise_pixels / total_pixels) * 100

# Function to detect noise type
def detect_noise_type(channel):
    mean = np.mean(channel)
    variance = np.var(channel)
    skewness = skew(channel.ravel())
    kurt = kurtosis(channel.ravel())

    # Check for Salt-and-Pepper Noise
    sp_noise = np.sum((channel == 0) | (channel == 255)) / channel.size
    if sp_noise > 0.005:  # If more than 0.5% of pixels are extreme
        return "Salt-and-Pepper Noise", sp_noise * 100

    # Check for Gaussian Noise (if variance is moderate)
    if 0.01 < variance < 100:
        return "Gaussian Noise", variance

    # Check for Speckle Noise (if variance is high and kurtosis is abnormal)
    if variance > 100 and kurt > 3:
        return "Speckle Noise", variance

    return "Unknown", 0

# Simulate a noisy image for testing (comment this out if using a real noisy image)
noisy_image = image.copy()
noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
noisy_image = cv2.add(noisy_image, noise)

# Detect noise type and percentage for each channel
for color, original, noisy in zip(["Red", "Green", "Blue"], [r, g, b], cv2.split(noisy_image)):
    noise_type, noise_value = detect_noise_type(noisy)
    noise_percent = noise_percentage(original, noisy)
    print(f"{color} Channel:")
    print(f"  Detected Noise Type: {noise_type}")
    print(f"  Noise Percentage: {noise_percent:.2f}%")
    print(f"  Noise Strength: {noise_value:.2f}\n")

# Display the original and noisy images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(noisy_image)
plt.title("Noisy Image")
plt.axis("off")

plt.show()
