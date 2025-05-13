import cv2
import numpy as np
import pickle
import os
from scipy.integrate import solve_ivp

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

def dna_to_binary(dna_str):
    mapping = {"A": "00", "T": "01", "C": "10", "G": "11"}
    return "".join(mapping[ch] for ch in dna_str)

def binary_to_dna(binary_str):
    mapping = {"00": "A", "01": "T", "10": "C", "11": "G"}
    return "".join(mapping[binary_str[i:i+2]] for i in range(0, len(binary_str), 2))

def dna_complement(dna_str):
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join(complement[ch] for ch in dna_str)

def reverse_dna_operations(img):
    rows, cols, channels = img.shape
    restored_img = np.zeros_like(img)

    for i in range(rows):
        for j in range(cols):
            for k in range(3):  
                binary_val = format(img[i, j, k], '08b')
                dna_seq = binary_to_dna(binary_val)
                original_dna = dna_complement(dna_seq)
                original_binary = dna_to_binary(original_dna)
                restored_img[i, j, k] = int(original_binary, 2)
    
    return restored_img

def xor_undiffuse(img, key_seq):
    flat_img = img.flatten()
    key_seq = (key_seq[:flat_img.size] * 256).astype(np.uint8)
    key_seq = np.resize(key_seq, flat_img.shape)  
    undiffused = flat_img ^ key_seq
    return undiffused.reshape(img.shape)

def inverse_permute_pixels(img):
    if not os.path.exists(PERMUTED_INDEX_PATH):
        print("Error: Permutation index file not found.")
        return img

    with open(PERMUTED_INDEX_PATH, "rb") as f:
        permuted_idx = pickle.load(f)

    rows, cols, channels = img.shape
    flat_img = img.reshape(-1, 3)
    inverse_permuted_img = np.zeros_like(flat_img)
    inverse_permuted_img[permuted_idx] = flat_img

    return inverse_permuted_img.reshape(img.shape)

def decrypt_image(enc_img_path, save_path):
    if not os.path.exists(enc_img_path):
        print(f"Error: Encrypted image '{enc_img_path}' not found.")
        return
    
    img = cv2.imread(enc_img_path)
    if img is None:
        print("Error: Failed to read the encrypted image.")
        return

    rows, cols, channels = img.shape
    key_seq = generate_chaotic_sequence(rows * cols * channels)
    key_seq = np.resize(key_seq, (rows * cols * channels))  

    dna_restored_img = reverse_dna_operations(img)

    undiffused_img = np.zeros_like(dna_restored_img)
    for i in range(3):
        undiffused_img[:, :, i] = xor_undiffuse(dna_restored_img[:, :, i], key_seq[i::3])

    original_img = inverse_permute_pixels(undiffused_img)

    cv2.imwrite(save_path, original_img)
    print(f"Decrypted image saved at: {os.path.abspath(save_path)}")

decrypt_image("encrypted.png", "decrypted.png")
