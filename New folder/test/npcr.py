import cv2
import numpy as np

def calculate_npcr_uaci(original, encrypted):
    # Split RGB channels
    orig_b, orig_g, orig_r = cv2.split(original)
    enc_b, enc_g, enc_r = cv2.split(encrypted)

    # Function to compute NPCR & UACI for a single channel
    def compute_channel_metrics(orig_channel, enc_channel):
        H, W = orig_channel.shape
        
        # NPCR Calculation
        D = (orig_channel != enc_channel).astype(np.uint8)
        NPCR = np.sum(D) / (H * W) * 100

        # UACI Calculation
        UACI = np.sum(np.abs(orig_channel - enc_channel) / 255) / (H * W) * 100

        return NPCR, UACI

    # Compute for R, G, B channels
    npcr_r, uaci_r = compute_channel_metrics(orig_r, enc_r)
    npcr_g, uaci_g = compute_channel_metrics(orig_g, enc_g)
    npcr_b, uaci_b = compute_channel_metrics(orig_b, enc_b)

    return (npcr_r, uaci_r), (npcr_g, uaci_g), (npcr_b, uaci_b)

# Load images
original_image = cv2.imread('application\src\Sample\Lenna.png')
encrypted_image = cv2.imread('encrypted.png')

# Compute NPCR and UACI for each channel
(r_npcr, r_uaci), (g_npcr, g_uaci), (b_npcr, b_uaci) = calculate_npcr_uaci(original_image, encrypted_image)

# Print results with 4 decimal points
print("NPCR & UACI for R, G, B Channels:")
print(f"Red   Channel - NPCR: {r_npcr:.4f}, UACI: {r_uaci:.4f}")
print(f"Green Channel - NPCR: {g_npcr:.4f}, UACI: {g_uaci:.4f}")
print(f"Blue  Channel - NPCR: {b_npcr:.4f}, UACI: {b_uaci:.4f}")
