import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy

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
original_img = cv2.imread(r"encrypted.png")
decrypted_img = cv2.imread(r"decrypted.png")

if original_img is None or decrypted_img is None:
    print("Error: One or both images could not be loaded. Check file paths.")
elif original_img.shape != decrypted_img.shape:
    print("Error: Image dimensions do not match!")
else:
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
