import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def correlation_3D(img):
    height, width, _ = img.shape
    img = img.astype(np.float64)  

    results = {}

    for i, color in enumerate(['Red', 'Green', 'Blue']):  
        x_h = img[:, :-1, i].flatten() 
        y_h = img[:, 1:, i].flatten()   
        corr_h = np.corrcoef(x_h, y_h)[0, 1]

        x_v = img[:-1, :, i].flatten()  
        y_v = img[1:, :, i].flatten()   
        corr_v = np.corrcoef(x_v, y_v)[0, 1]

        x_d = img[:-1, :-1, i].flatten()  
        y_d = img[1:, 1:, i].flatten()   
        corr_d = np.corrcoef(x_d, y_d)[0, 1]

        results[color] = {
            "horizontal": (x_h, y_h, corr_h),
            "vertical": (x_v, y_v, corr_v),
            "diagonal": (x_d, y_d, corr_d)
        }

        
        print(f"Correlation Coefficients for {color} Channel:")
        print(f"  Horizontal: {corr_h:.6f}")
        print(f"  Vertical  : {corr_v:.6f}")
        print(f"  Diagonal  : {corr_d:.6f}")
        print("-" * 40)

    return results

def plot_3D(x_h, y_h, x_v, y_v, x_d, y_d, title, color, ax):
    ax.scatter(x_h[:5000], y_h[:5000], x_v[:5000], s=1, alpha=0.5, color=color)  # Scatter plot for first 5000 points
    ax.set_title(title)
    ax.set_xlabel("Horizontal")
    ax.set_ylabel("Vertical")
    ax.set_zlabel("Diagonal")

image = cv2.imread("application\src\output\encrypted.png")  


results = correlation_3D(image)

fig = plt.figure(figsize=(15, 10))

colors = {"Red": "red", "Green": "green", "Blue": "blue"}
positions = [(1, "(a) Red"), (2, "(b) Green"), (3, "(c) Blue")]

for i, (color, pos) in enumerate(zip(results.keys(), positions)):
    ax = fig.add_subplot(2, 3, pos[0], projection='3d')
    data = results[color]
    plot_3D(*data["horizontal"][:2], *data["vertical"][:2], *data["diagonal"][:2], pos[1], colors[color], ax)

plt.tight_layout()
plt.show()
