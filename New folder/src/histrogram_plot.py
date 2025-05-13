import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histograms(original_image_path, encrypted_image_path):
    # Load images
    original = cv2.imread(original_image_path)
    encrypted = cv2.imread(encrypted_image_path)

    if original is None or encrypted is None:
        print("Error: Could not load images. Check file paths!")
        return

    # Convert images to RGB (OpenCV loads in BGR format)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    encrypted = cv2.cvtColor(encrypted, cv2.COLOR_BGR2RGB)

    # Split channels (R, G, B)
    channels = ['Red', 'Green', 'Blue']
    colors = ['r', 'g', 'b']

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    fig.suptitle("Histogram Comparison: Original vs Encrypted Image", fontsize=14)

    for i, (color, channel) in enumerate(zip(colors, channels)):
        # Compute histogram
        hist_orig = cv2.calcHist([original], [i], None, [256], [0, 256])
        hist_enc = cv2.calcHist([encrypted], [i], None, [256], [0, 256])

        # Plot Original Image Histogram
        axes[0, i].plot(hist_orig, color=color)
        axes[0, i].set_title(f'Original {channel} Channel')
        axes[0, i].set_xlim([0, 256])

        # Plot Encrypted Image Histogram
        axes[1, i].plot(hist_enc, color=color)
        axes[1, i].set_title(f'Encrypted {channel} Channel')
        axes[1, i].set_xlim([0, 256])

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

# Example usage
plot_histograms("application/src/Sample/Lenna.png", "application\src\output\encrypted.png")
