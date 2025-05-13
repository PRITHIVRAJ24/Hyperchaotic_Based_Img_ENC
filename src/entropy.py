import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


image = cv2.imread("application\Chen-Lorenz\Encrypted.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


r, g, b = cv2.split(image)


def compute_entropy(channel):
    hist, _ = np.histogram(channel.ravel(), bins=256, range=[0,256], density=True)
    return entropy(hist, base=2)


metrics = {}
for color, channel in zip(["Red", "Green", "Blue"], [r, g, b]):
    
    channel_entropy = compute_entropy(channel)
    
    metrics[color] = {
     
        "Entropy": channel_entropy,
      
    }
base = np.random.normal(0, 25, image.shape).astype(np.uint8)
en_image = cv2.add(image, base)


for color in ["Red", "Green", "Blue"]:
    print(f"{color} Channel:")
    
    print(f"  Entropy: {metrics[color]['Entropy']}")

