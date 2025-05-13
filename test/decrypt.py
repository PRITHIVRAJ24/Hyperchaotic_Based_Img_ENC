import cv2
import numpy as np
import os
import pickle
from encrypt import generate_chaotic_sequence, mse, psnr

PERMUTED_INDEX_PATH = "permuted_idx.pkl"

def inverse_permute(img, permuted_idx):
    rows, cols, channels = img.shape
    flat_img = img.reshape(-1, 3)
    recovered_img = np.zeros_like(flat_img)
    recovered_img[permuted_idx] = flat_img
    return recovered_img.reshape(img.shape)

def inverse_xor_diffuse(img, key_seq):
    flat_img = img.flatten()
    key_seq = (key_seq[:flat_img.size] * 256).astype(np.uint8)
    recovered = np.bitwise_xor(flat_img, key_seq)  # Ensure proper bitwise XOR
    return recovered.reshape(img.shape)

def decrypt_image(enc_path, save_path, original_img_path):
    if not os.path.exists(enc_path):
        print(f"Error: Encrypted image '{enc_path}' not found.")
        return
    if not os.path.exists(original_img_path):
        print(f"Error: Original image '{original_img_path}' not found.")
        return
    if not os.path.exists(PERMUTED_INDEX_PATH):
        print(f"Error: Permuted index file '{PERMUTED_INDEX_PATH}' not found.")
        return

    enc_img = cv2.imread(enc_path)
    original_img = cv2.imread(original_img_path)
    if enc_img is None or original_img is None:
        print("Error: Failed to read encrypted or original image.")
        return

    with open(PERMUTED_INDEX_PATH, "rb") as f:
        permuted_idx = pickle.load(f)

    rows, cols, channels = enc_img.shape
    key_seq = generate_chaotic_sequence(rows * cols * channels)
    key_seq = np.resize(key_seq, (rows * cols * channels))

   
    undiffused_img = np.zeros_like(enc_img)
    for i in range(3):
        undiffused_img[:, :, i] = inverse_xor_diffuse(enc_img[:, :, i], key_seq)

  
    recovered_img = inverse_permute(undiffused_img, permuted_idx)

    cv2.imwrite(save_path, recovered_img)
    dec_mse = mse(original_img, recovered_img)
    dec_psnr = psnr(original_img, recovered_img)
    print(f"Decrypted Image - MSE: {dec_mse}, PSNR: {dec_psnr}")
    print(f"Decrypted image saved at: {os.path.abspath(save_path)}")

decrypt_image("application/src/output/noise_encrytped.png", "application/src/output/noise_decrypted.png", "application/src/Sample/Lenna.png")
