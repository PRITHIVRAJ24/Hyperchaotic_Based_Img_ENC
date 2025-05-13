import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import random

def add_salt_and_pepper_noise(image, amount=0.02):
    """Adds salt and pepper noise to an image."""
    noisy_image = image.copy()
    num_pixels = int(amount * image.size)

    for _ in range(num_pixels):
        x, y = random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[0] - 1)
        noisy_image[y, x] = 0 if random.random() < 0.5 else 255  # Salt and pepper

    return noisy_image

def add_gaussian_noise(image, mean=0, var=10):
    """Adds Gaussian noise to an image."""
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gaussian)
    return np.clip(noisy_image, 0, 255)

def calculate_mse(original, decrypted):
    mse_values = []
    for i in range(3):
        mse = np.mean((original[:, :, i] - decrypted[:, :, i]) ** 2)
        mse_values.append(mse)
    return np.mean(mse_values)

def calculate_psnr(original, decrypted):
    mse = calculate_mse(original, decrypted)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)

def calculate_ssim(original, decrypted):
    ssim_values = []
    for i in range(3):
        ssim_value = ssim(original[:, :, i], decrypted[:, :, i], data_range=255)
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)

def plot_histogram_with_images(original, decrypted):
    colors = ('b', 'g', 'r')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Display original image
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Display decrypted image
    axes[0, 1].imshow(cv2.cvtColor(decrypted, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Decrypted Image")
    axes[0, 1].axis("off")

    # Plot histograms for original image
    for i, color in enumerate(colors):
        hist_orig = cv2.calcHist([original], [i], None, [256], [0, 256])
        axes[1, 0].plot(hist_orig, color=color)
    axes[1, 0].set_title("Original Image Histogram")
    axes[1, 0].set_xlim([0, 256])

    # Plot histograms for decrypted image
    for i, color in enumerate(colors):
        hist_decr = cv2.calcHist([decrypted], [i], None, [256], [0, 256])
        axes[1, 1].plot(hist_decr, color=color, linestyle='dashed')
    axes[1, 1].set_title("Decrypted Image Histogram")
    axes[1, 1].set_xlim([0, 256])

    plt.tight_layout()
    plt.show()

def calculate_entropy(image):
    """Computes entropy for each channel and returns the mean entropy."""
    entropy_values = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()  # Normalize histogram
        entropy_values.append(entropy(hist + 1e-10))  # Avoid log(0)
    return np.mean(entropy_values)

# Load images with corrected paths
original_img = cv2.imread(r"application/src/Sample/Lenna.png")
decrypted_img = cv2.imread(r"application/src/output/encrypted_test.png")

if original_img is None or decrypted_img is None:
    print("Error: One or both images could not be loaded. Check file paths.")
elif original_img.shape != decrypted_img.shape:
    print("Error: Image dimensions do not match!")
else:
    # Adding noise to test robustness
    salt_pepper_noisy_img = add_salt_and_pepper_noise(original_img, amount=0.02)
    gaussian_noisy_img = add_gaussian_noise(original_img, mean=0, var=10)

    mse_value = calculate_mse(original_img, decrypted_img)
    psnr_value = calculate_psnr(original_img, decrypted_img)
    ssim_value = calculate_ssim(original_img, decrypted_img)
    entropy_original = calculate_entropy(original_img)
    entropy_decrypted = calculate_entropy(decrypted_img)

    print(f"Mean Squared Error (MSE): {mse_value:.4f}")
    print(f"PSNR Value: {psnr_value:.2f} dB")
    print(f"SSIM Value: {ssim_value:.4f}")
    print(f"Entropy (Original): {entropy_original:.4f}")
    print(f"Entropy (Decrypted): {entropy_decrypted:.4f}")

    plot_histogram_with_images(original_img, decrypted_img)

    # Display noisy images
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(cv2.cvtColor(salt_pepper_noisy_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Salt & Pepper Noisy Image")
    ax[0].axis("off")

    ax[1].imshow(cv2.cvtColor(gaussian_noisy_img, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Gaussian Noisy Image")
    ax[1].axis("off")

    plt.show()
