import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded image
# image_path = 'images\\capture_20250429_184911.png'
image_path ='images\\right_angle\\1.png'
image = cv2.imread(image_path)
output = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using Hough Circle Transform
circles = cv2.HoughCircles(
    gray_blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=100,
    param1=100,
    param2=100,
    minRadius=100,
    maxRadius=200
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    # Define cropping box (ensure it's within image bounds)
    x1 = max(x - r, 0)
    y1 = max(y - r, 0)
    x2 = min(x + r, image.shape[1])
    y2 = min(y + r, image.shape[0])

    cropped = image[y1:y2, x1:x2]

    # Resize to 200x200
    cropped_resized = cv2.resize(cropped, (200, 200))

# Convert BGR to RGB for display
cropped_rgb = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB)

# Display the cropped and resized circle
plt.figure(figsize=(4, 4))
plt.imshow(cropped_rgb)
plt.title("Cropped & Resized Circle (200x200)")
plt.axis("off")



# Create a black background image (200x200 like the resized circle)
mask = np.ones((200, 200), dtype=np.uint8)

# Draw white filled circle at center (100, 100) with radius 50
cv2.circle(mask, (100, 100), 50, 255, -1)

# Apply the mask: keep the circle area, set the rest to black
masked_circle = cv2.bitwise_and(cropped_resized, cropped_resized, mask=mask)

# Display result
masked_rgb = cv2.cvtColor(masked_circle, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(4, 4))
plt.imshow(masked_rgb)
plt.title("Circle Mask at (100,100), R=50")
plt.axis("off")


# Create a white background
white_background = np.ones_like(cropped_resized, dtype=np.uint8) * 255

# Invert the mask so the outer part is 255 (white), inner part is 0
inverted_mask = cv2.bitwise_not(mask)

# Apply the inverted mask to the white background
background = cv2.bitwise_and(white_background, white_background, mask=inverted_mask)

# Combine the masked circle and white background
final_image = cv2.add(masked_circle, background)

# Display the result
final_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(4, 4))
plt.imshow(final_rgb)
plt.title("Circle on White Background")
plt.axis("off")
# Convert the white-background image to grayscale
grayscale_result = cv2.cvtColor(final_rgb, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.figure(figsize=(4, 4))
plt.imshow(grayscale_result, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

# Apply binary threshold to the grayscale image
_, thresholded_image = cv2.threshold(grayscale_result, 127, 255, cv2.THRESH_BINARY)

# Display the thresholded image
plt.figure(figsize=(4, 4))
plt.imshow(thresholded_image, cmap='gray')
plt.title("Binary Thresholded Image")
plt.axis("off")

# Convert thresholded image to 3-channel format
thresholded_3ch = cv2.merge([thresholded_image] * 3)

# Wherever the thresholded image is white (255), set color image to white
combined_image = final_image.copy()
combined_image[thresholded_3ch == 255] = 255  # Set to white

# Display the updated color image
combined_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(4, 4))
plt.imshow(combined_rgb)
plt.title("Color Image with White Binary Regions")
plt.axis("off")



image_np = np.array(combined_rgb)

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
cv2.circle(red_mask, center=(100, 100), radius=20, color=(0, 0, 0), thickness=-1)
cv2.circle(black_mask, center=(100, 100), radius=20, color=(0, 0, 0), thickness=-1)
plt.figure(figsize=(4, 4))
plt.title("Red Mask")
plt.imshow(red_mask, cmap='gray')
plt.axis("off")

plt.figure(figsize=(4, 4))
plt.title("Black Mask")
plt.imshow(black_mask, cmap='gray')
plt.axis("off")

plt.tight_layout()

# Threshold the image to ensure it's binary
_, binary_image = cv2.threshold(black_mask, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an image to draw the fitted lines
fitted_line_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

# Fit lines to contours and draw
for cnt in contours:
    if cv2.contourArea(cnt) > 20:  # Filter out small contours
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        # Compute line endpoints for drawing
        lefty = int((-x * vy / vx) + y)
        righty = int(((binary_image.shape[1] - x) * vy / vx) + y)
        cv2.line(fitted_line_image, (binary_image.shape[1] - 1, righty), (0, lefty), (0, 255, 0), 2)

# Display the result
plt.figure(figsize=(4, 4))
plt.title("Contours with Fitted Lines")
plt.imshow(fitted_line_image)
plt.axis("off")
plt.show()

import math

# Store angles
angles = []

# Fit lines and compute angles
