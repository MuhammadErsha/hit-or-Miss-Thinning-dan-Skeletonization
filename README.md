# hit-or-Miss-Thinning-dan-Skeletonization
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Upload gambar tulisan terlebih dahulu (gunakan sidebar di Colab)
from google.colab import files
uploaded = files.upload()

# Baca gambar (asumsinya file PNG)
file_name = list(uploaded.keys())[0]
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Definisi 9 structuring elements
kernels = {
    "Rect 3x3": cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    "Rect 5x5": cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    "Ellipse 3x3": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    "Ellipse 5x5": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    "Cross 3x3": cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
    "Cross 5x5": cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
    "Line H 9x1": cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),
    "Line V 1x9": cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)),
    "Diagonal 5x5": np.eye(5, dtype=np.uint8)
}

# Proses erosi dan dilasi
erosions = {}
dilations = {}

for name, kernel in kernels.items():
    erosions[name] = cv2.erode(binary, kernel, iterations=1)
    dilations[name] = cv2.dilate(binary, kernel, iterations=1)

# Tampilkan hasil
fig, axs = plt.subplots(len(kernels), 3, figsize=(12, 25))
fig.suptitle('Erosi & Dilasi dengan 9 Structuring Elements', fontsize=16)

for i, name in enumerate(kernels):
    axs[i, 0].imshow(binary, cmap='gray')
    axs[i, 0].set_title('Original')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(erosions[name], cmap='gray')
    axs[i, 1].set_title(f'Erosion\n{name}')
    axs[i, 1].axis('off')

    axs[i, 2].imshow(dilations[name], cmap='gray')
    axs[i, 2].set_title(f'Dilation\n{name}')
    axs[i, 2].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
