import cv2
import numpy as np
from scipy.integrate import solve_ivp
import os

INPUT_IMAGE_PATH = 'application\src\Sample\Lenna.png'
ENCRYPTED_IMAGE_PATH = 'encrypted.png'
DECRYPTED_IMAGE_PATH = 'decrypted.png'

# Chen chaotic system
def chen_system(t, state, a=35, b=3, c=28):
    x, y, z = state
    dxdt = a * (y - x)
    dydt = (c - a) * x - x * z + c * y
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]

def generate_chaotic_sequence(length, initial_conditions=[0.1, 0.2, 0.3]):
    t_span = (0, length * 0.01)
    t_eval = np.linspace(*t_span, length)
    sol = solve_ivp(chen_system, t_span, initial_conditions, t_eval=t_eval, method='RK45')
    return np.mod(np.abs(sol.y[0]), 1)  

def permute_pixels(img, key_seq):
    rows, cols, channels = img.shape
    flat_img = img.reshape(-1, 3)
    permuted_idx = np.argsort(key_seq[:rows * cols])
    return flat_img[permuted_idx].reshape(img.shape), permuted_idx

def inverse_permute(img, permuted_idx):
    rows, cols, channels = img.shape
    flat_img = img.reshape(-1, 3)
    recovered_img = np.zeros_like(flat_img)
    recovered_img[permuted_idx] = flat_img
    return recovered_img.reshape(img.shape)

def xor_diffuse(img, key_seq):
    flat_img = img.flatten()
    key_seq = (key_seq[:flat_img.size] * 256).astype(np.uint8)
    diffused = flat_img ^ key_seq
    return diffused.reshape(img.shape)

def inverse_xor_diffuse(img, key_seq):
    return xor_diffuse(img, key_seq)  # XOR is its own inverse

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err

def psnr(imageA, imageB):
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return 100  
    return 10 * np.log10((255 ** 2) / mse_value)

# Encryption
def encrypt_image(img_path, save_path):
    if not os.path.exists(img_path):
        print(f"Error: Input image '{img_path}' not found.")
        return None
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Failed to read the input image. Check the file format and path.")
        return None

    rows, cols, channels = img.shape
    total_pixels = rows * cols * channels
    key_seq = generate_chaotic_sequence(total_pixels)

    permuted_img, permuted_idx = permute_pixels(img, key_seq)
    diffused_img = np.zeros_like(permuted_img)

    for i in range(3):
        diffused_img[:, :, i] = xor_diffuse(permuted_img[:, :, i], key_seq)

    cv2.imwrite(save_path, diffused_img)
    enc_mse = mse(img, diffused_img)
    enc_psnr = psnr(img, diffused_img)
    print(f"Encrypted Image - MSE: {enc_mse}, PSNR: {enc_psnr}")
    print(f"Encrypted image saved at: {os.path.abspath(save_path)}")
    
    return permuted_idx  


def decrypt_image(enc_path, save_path, permuted_idx, original_img_path):
    if not os.path.exists(enc_path):
        print(f"Error: Encrypted image '{enc_path}' not found.")
        return
    if not os.path.exists(original_img_path):
        print(f"Error: Original image '{original_img_path}' not found.")
        return

    enc_img = cv2.imread(enc_path)
    original_img = cv2.imread(original_img_path)
    if enc_img is None or original_img is None:
        print("Error: Failed to read encrypted or original image.")
        return

    rows, cols, channels = enc_img.shape
    total_pixels = rows * cols * channels
    key_seq = generate_chaotic_sequence(total_pixels)  # Regenerate key sequence

    undiffused_img = np.zeros_like(enc_img)
    for i in range(3):
        undiffused_img[:, :, i] = inverse_xor_diffuse(enc_img[:, :, i], key_seq)

    recovered_img = inverse_permute(undiffused_img, permuted_idx)

    cv2.imwrite(save_path, recovered_img)
    dec_mse = mse(original_img, recovered_img)
    dec_psnr = psnr(original_img, recovered_img)
    print(f"Decrypted Image - MSE: {dec_mse}, PSNR: {dec_psnr}")
    print(f"Decrypted image saved at: {os.path.abspath(save_path)}")

perm_idx = encrypt_image(INPUT_IMAGE_PATH, ENCRYPTED_IMAGE_PATH)
if perm_idx is not None:
    decrypt_image(ENCRYPTED_IMAGE_PATH, DECRYPTED_IMAGE_PATH, perm_idx, INPUT_IMAGE_PATH)
