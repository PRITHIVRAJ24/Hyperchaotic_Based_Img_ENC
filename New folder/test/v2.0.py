import cv2
import numpy as np
from scipy.integrate import solve_ivp
import os
import pickle
from numba import njit

PERMUTED_INDEX_PATH = "permuted_idx.pkl"

def chen_hyperchaotic_system(t, state, a=35, b=3, c=28, d=10, e=5):
    x, y, z, w = state
    return [a * (y - x) + w, (c - a) * x - x * z + c * y, x * y - b * z, -d * x + e * w]

def generate_hyperchaotic_sequence(length, initial_conditions=[0.1, 0.2, 0.3, 0.4]):
    t_eval = np.linspace(0, length * 0.01, length)
    sol = solve_ivp(chen_hyperchaotic_system, (0, length * 0.01), initial_conditions, t_eval=t_eval, method='RK45')
    return np.mod(np.abs(sol.y[:4].sum(axis=0)), 1)  # Use sum of 4D variables for stronger key
def permute_pixels(img, key_seq):
    permuted_idx = np.argsort(key_seq[:img.size // 3])
    with open(PERMUTED_INDEX_PATH, "wb") as f:
        pickle.dump(permuted_idx, f)
    return img.reshape(-1, 3)[permuted_idx].reshape(img.shape)

@njit  # Optimize XOR operation with Numba
def xor_diffuse(img_flat, key_seq):
    return img_flat ^ key_seq

def dna_encode(img):
    dna_rules = np.array([0b00, 0b01, 0b10, 0b11], dtype=np.uint8)  # DNA encoding rules
    return np.unpackbits(img, axis=-1)[:, ::2] @ dna_rules  # Convert binary to DNA values

def dna_xor(img, key_seq):
    img_dna = dna_encode(img)
    key_dna = dna_encode(key_seq)
    return np.packbits(img_dna ^ key_dna, axis=-1)

def mse(imageA, imageB):
    return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    return 100 if mse_value == 0 else 10 * np.log10((255 ** 2) / mse_value)

def encrypt_image(img_path, save_path):
    if not os.path.exists(img_path):
        print(f"Error: Input image '{img_path}' not found.")
        return
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Failed to read the input image.")
        return
    key_seq = (generate_hyperchaotic_sequence(img.size) * 256).astype(np.uint8)
    permuted_img = permute_pixels(img, key_seq)
    diffused_img = dna_xor(permuted_img, key_seq)  # Use DNA XOR operation
    cv2.imwrite(save_path, diffused_img)
    print(f"Encrypted Image - MSE: {mse(img, diffused_img)}, PSNR: {psnr(img, diffused_img)}")
    print(f"Encrypted image saved at: {os.path.abspath(save_path)}")

encrypt_image(os.path.join("application", "src", "Sample", "Lenna.png"), "encrypted.png")
