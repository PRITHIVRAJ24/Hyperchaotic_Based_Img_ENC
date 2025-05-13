import cv2
import numpy as np
from scipy.integrate import solve_ivp
import os
import pickle

PERMUTED_INDEX_PATH = "permuted_idx.pkl"

def chen_system(t, state, a=36, b=3, c=28, d=0.7, r=0.5):
    x, y, z, w = state
    return np.array([
        a * (y - x),
        (c - a) * x - x * z + c * y,
        x * y - b * z,
        -d * z + r * w
    ], dtype=np.float64)

def generate_chaotic_sequence(length, initial_conditions=[0.1, 0.2, 0.3, 0.4]):
    t_span = (0, length * 0.01)
    t_eval = np.linspace(*t_span, length)
    sol = solve_ivp(chen_system, t_span, initial_conditions, t_eval=t_eval, method='RK45')
    return np.mod(np.abs(sol.y[0]), 1)

def permute_pixels(img, key_seq):
    rows, cols, channels = img.shape
    flat_img = img.reshape(-1, 3)
    permuted_idx = np.argsort(key_seq[:rows * cols])
    
    permuted_img = np.zeros_like(flat_img)
    permuted_img[permuted_idx] = flat_img
    
    with open(PERMUTED_INDEX_PATH, "wb") as f:
        pickle.dump(permuted_idx, f)
    
    return permuted_img.reshape(img.shape)

# DNA Mapping
def binary_to_dna(binary_str):
    mapping = {"00": "A", "01": "T", "10": "C", "11": "G"}
    return "".join(mapping[binary_str[i:i+2]] for i in range(0, len(binary_str), 2))

def dna_to_binary(dna_str):
    mapping = {"A": "00", "T": "01", "C": "10", "G": "11"}
    return "".join(mapping[ch] for ch in dna_str)

def dna_complement(dna_str):
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(complement[ch] for ch in dna_str)

def apply_dna_operations(img):
    rows, cols, channels = img.shape
    modified_img = np.zeros_like(img)
    
    for i in range(rows):
        for j in range(cols):
            for k in range(3):  # R, G, B
                binary_val = format(img[i, j, k], '08b')
                dna_seq = binary_to_dna(binary_val)
                complemented_dna = dna_complement(dna_seq)
                modified_binary = dna_to_binary(complemented_dna)
                modified_img[i, j, k] = int(modified_binary, 2)
    
    return modified_img

def xor_diffuse(img, key_seq):
    flat_img = img.flatten()
    key_seq = (key_seq[:flat_img.size] * 256).astype(np.uint8)
    key_seq = np.resize(key_seq, flat_img.shape)  # Ensure key matches image size
    diffused = flat_img ^ key_seq
    return diffused.reshape(img.shape)

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err

def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return 100  
    return 10 * np.log10((255 ** 2) / mse_value)

def encrypt_image(img_path, save_path):
    if not os.path.exists(img_path):
        print(f"Error: Input image '{img_path}' not found.")
        return
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Failed to read the input image.")
        return

    rows, cols, channels = img.shape
    key_seq = generate_chaotic_sequence(rows * cols * channels)
    key_seq = np.resize(key_seq, (rows * cols * channels))  # Ensure correct length
    
    permuted_img = permute_pixels(img, key_seq[:rows * cols])
    
    diffused_img = np.zeros_like(permuted_img)
    for i in range(3):
        diffused_img[:, :, i] = xor_diffuse(permuted_img[:, :, i], key_seq[i::3])
    
    dna_modified_img = apply_dna_operations(diffused_img)
    
    cv2.imwrite(save_path, dna_modified_img)
    enc_mse = mse(img, dna_modified_img)
    enc_psnr = psnr(img, dna_modified_img)
    print(f"MSE: {enc_mse}, PSNR: {enc_psnr}")
    print(f"Encrypted image saved at: {os.path.abspath(save_path)}")

encrypt_image("application/src/Sample/Lenna.png", "encrypted.png")