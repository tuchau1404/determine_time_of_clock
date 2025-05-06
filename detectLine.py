
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
# Load the image
image_path='images\\Figure_6.png'
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Convert to HSV for better color segmentation
hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

# Define red color range in HSV (two ranges to cover red hue wrap-around)
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 70, 50])
upper_red2 = np.array([180, 255, 255])

# Define black color range in HSV
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# Create masks
red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(red_mask1, red_mask2)

black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
cv2.circle(red_mask, center=(200, 200), radius=40, color=(0, 0, 0), thickness=-1)
cv2.circle(black_mask, center=(200, 200), radius=40, color=(0, 0, 0), thickness=-1)
# Display red and black masks separately
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title("Red Mask")
plt.imshow(red_mask, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Black Mask")
plt.imshow(black_mask, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()