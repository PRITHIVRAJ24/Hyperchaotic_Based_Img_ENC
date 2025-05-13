import cv2
import numpy as np
from PIL import Image

def mse(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")
    err = np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2, axis=(0, 1))
    return err

def psnr(image1, image2):
    mse_values = mse(image1, image2)
    if np.any(mse_values == 0):
        return float('inf')
    max_pixel = 255.0
    psnr_values = 20 * np.log10(max_pixel / np.sqrt(mse_values))
    return psnr_values

def chen_system(image, a=100, b_const=10, c=80, step=0.01, iterations=1):
    r, g, b = cv2.split(image.astype(np.float64))
    for _ in range(iterations):
        dx = a * (g - r)
        dy = c * r - g - r * b
        dz = b_const * b + r * g
        r = np.clip(r + dx * step, 0, 255)
        g = np.clip(g + dy * step, 0, 255)
        b = np.clip(b + dz * step, 0, 255)
    return cv2.merge((r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)))

def inverse_chen_system(image, a=100, b_const=10, c=80, step=0.01, iterations=1):
    r, g, b = cv2.split(image.astype(np.float64))
    for _ in range(iterations):
        dx = a * (g - r)
        dy = c * r - g - r * b
        dz = b_const * b + r * g
        r = np.clip(r - dx * step, 0, 255)
        g = np.clip(g - dy * step, 0, 255)
        b = np.clip(b - dz * step, 0, 255)
    return cv2.merge((r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)))

def dna_mapping(image, key=54321):
    np.random.seed(key)
    mask = np.random.randint(0, 256, image.shape, dtype=np.uint8)
    return cv2.bitwise_xor(image, mask)

def inverse_dna_mapping(image, key=54321):
    return dna_mapping(image, key)  # XOR operation is reversible

def pixel_diffusion(image, seed=12345):
    np.random.seed(seed)
    indices = np.arange(image.size)
    np.random.shuffle(indices)
    shuffled_image = np.zeros_like(image.flatten())
    shuffled_image[indices] = image.flatten()
    return shuffled_image.reshape(image.shape), indices

def inverse_pixel_diffusion(image, indices):
    original_image = np.zeros_like(image.flatten())
    original_image[indices] = image.flatten()
    return original_image.reshape(image.shape)

def encrypt_image(file_path, output_path):
    image = Image.open(file_path).convert('RGB')
    image = np.array(image)
    image = chen_system(image)
    image = dna_mapping(image)
    shuffled_image, indices = pixel_diffusion(image)
    np.save(output_path + "_indices.npy", indices)  # Save shuffling indices
    Image.fromarray(shuffled_image).save(output_path)
    print(f"Encrypted image saved at: {output_path}")

def decrypt_image(file_path, indices_path, output_path):
    encrypted_image = Image.open(file_path).convert('RGB')
    encrypted_image = np.array(encrypted_image)
    indices = np.load(indices_path)
    decrypted_image = inverse_pixel_diffusion(encrypted_image, indices)
    decrypted_image = inverse_dna_mapping(decrypted_image)
    decrypted_image = inverse_chen_system(decrypted_image)
    Image.fromarray(decrypted_image).save(output_path)
    print(f"Decrypted image saved at: {output_path}")

# Encrypt image
encrypt_image("application/src/Sample/Lenna.png", "application/src/output/encrypted_test.png")

# Decrypt image
decrypt_image("application/src/output/encrypted_test.png", "application/src/output/encrypted_test.png_indices.npy", "application/src/output/decrypted_test.png")

# Load images
original = cv2.imread("application/src/Sample/Lenna.png")
decrypted = cv2.imread("application/src/output/decrypted_test.png")

# Ensure images are loaded
if original is None or decrypted is None:
    raise FileNotFoundError("Error: One or both images could not be loaded.")

# Convert to same datatype before processing
original = original.astype(np.uint8)
decrypted = decrypted.astype(np.uint8)

mse_values = mse(original, decrypted)
psnr_values = psnr(original, decrypted)

print(f"MSE: {mse_values}")
print(f"PSNR: {psnr_values} dB")
