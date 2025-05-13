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



def inverse_chen_system(image, a=35, b_const=3, c=28, step=0.01, iterations=1):
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
    return dna_mapping(image, key)



def inverse_pixel_diffusion(image):
    indices = np.arange(image.size)
    np.random.shuffle(indices)
    inverse_indices = np.argsort(indices)
    return image.flatten()[inverse_indices].reshape(image.shape)



def decrypt_image(file_path, output_path):
    image = Image.open(file_path).convert('RGB')
    image = np.array(image)
    image = inverse_pixel_diffusion(image)
    image = inverse_dna_mapping(image)
    image = inverse_chen_system(image)
    Image.fromarray(image).save(output_path)
    print(f"Decrypted image saved at: {output_path}")




decrypt_image("application\src\output\encrypted_test.png", "application\src\output\decrypted_image.png")

# Load images
original = cv2.imread("application\src\Sample\Lenna.png")
decrypted = cv2.imread("application\src\output\decrypted_image.png")

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
