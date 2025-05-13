import cv2
import numpy as np

def mse(imageA, imageB):
    
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):
    
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf') 
    return 10 * np.log10((255 ** 2) / mse_value)


image = cv2.imread("application/src/output/noise_decrypted.png")
original_image = cv2.imread("application\src\Sample\Lenna.png")


filtered_image = np.zeros_like(image)
for i in range(3):  
    filtered_image[:, :, i] = cv2.medianBlur(image[:, :, i], 5) 

cv2.imwrite("application/src/output/filtered_image.png", filtered_image)


mse_value = mse(original_image, filtered_image)
psnr_value = psnr(original_image, filtered_image)


print(f"MSE: {mse_value}")
print(f"PSNR: {psnr_value} dB")

cv2.imshow("Median Filtered Image", filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
