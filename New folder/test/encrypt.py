import cv2
import numpy as np
from scipy.integrate import solve_ivp
import os
import pickle

PERMUTED_INDEX_PATH = "permuted_idx.pkl"
time=0
def chen_system(t, state, a=36, b=3, c=28 , d=0.7, r=0.57):
    global time
    x, y, z, w = state
    dxdt = a * (y - x)
    dydt = (c - a) * x - x * z + c * y
    dzdt = x * y - b * z
    dwdt = -d * z + r * w
    time+=1
    return [dxdt, dydt, dzdt,dwdt]

def generate_chaotic_sequence(length, initial_conditions=[0.1, 0.2, 0.3, 0.5]):
  
    t_span = (0, length * 0.01)
    t_eval = np.linspace(*t_span, length)
    sol = solve_ivp(chen_system, t_span, initial_conditions, t_eval=t_eval, method='RK45')
    return np.mod(np.abs(sol.y[0]), 1)

def permute_pixels(img, key_seq):
    rows, cols, channels = img.shape
    flat_img = img.reshape(-1, 3)
    permuted_idx = np.argsort(key_seq[:rows * cols])
    
    # Store permuted index in a file
    with open(PERMUTED_INDEX_PATH, "wb") as f:
        pickle.dump(permuted_idx, f)

    return flat_img[permuted_idx].reshape(img.shape)

def xor_diffuse(img, key_seq):
    flat_img = img.flatten()
    key_seq = (key_seq[:flat_img.size] * 256).astype(np.uint8)
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
    #total_pixels = rows * cols * channels
    #key_seq = generate_chaotic_sequence(total_pixels)
    key_seq = generate_chaotic_sequence(rows * cols * channels)
    key_seq = np.resize(key_seq, (rows * cols * channels)) 
    permuted_img = permute_pixels(img, key_seq)
    diffused_img = np.zeros_like(permuted_img)

    for i in range(3):
        diffused_img[:, :, i] = xor_diffuse(permuted_img[:, :, i], key_seq)

    cv2.imwrite(save_path, diffused_img)
    enc_mse = mse(img, diffused_img)
    enc_psnr = psnr(img, diffused_img)
    print (time)
    print(f"Encrypted Image - MSE: {enc_mse}, PSNR: {enc_psnr}")
    print(f"Encrypted image saved at: {os.path.abspath(save_path)}")
encrypt_image("application/src/Sample/Lenna.png","encrypted.png")