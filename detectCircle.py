# Re-import required libraries after reset
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reload the image
image_path = "images\\right_angle\\8_20.5.png"
image = cv2.imread(image_path)

# Convert to HSV for color thresholding
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV range for dark blue to nearly black
lower_dark_blue = np.array([100, 0, 0])
upper_dark_blue = np.array([150, 255, 120])

# Mask for dark blue range
mask_dark_blue = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)

# Apply mask to keep only dark blue parts
result_dark_blue = cv2.bitwise_and(image, image, mask=mask_dark_blue)

# Convert masked result to grayscale
gray_masked = cv2.cvtColor(result_dark_blue, cv2.COLOR_BGR2GRAY)

# Threshold to binary
_, binary_mask = cv2.threshold(gray_masked, 10, 255, cv2.THRESH_BINARY)
plt.figure(figsize=(6, 6))
plt.title("Thresholding")
plt.imshow(binary_mask, cmap='gray')

# Apply Hough Circle Transform on the binary mask
circles_masked = cv2.HoughCircles(
    binary_mask,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=100,
    param1=100,
    param2=30,
    minRadius=50,
    maxRadius=300
)

# Draw the detected largest circle and crop it
circle_detected = image.copy()
if circles_masked is not None:
    circles_masked = np.uint16(np.around(circles_masked))
    largest_circle = max(circles_masked[0, :], key=lambda c: c[2])
    x, y, r = largest_circle
    cv2.circle(circle_detected, (x, y), r, (0, 0, 255), 3)
    cv2.circle(circle_detected, (x, y), 2, (0, 255, 0), 3)

    # Crop image based on the circle
    x1, y1 = max(x - r, 0), max(y - r, 0)
    x2, y2 = min(x + r, image.shape[1]), min(y + r, image.shape[0])
    cropped_image = image[y1:y2, x1:x2]
    cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Show cropped image
    plt.figure(figsize=(6, 6))
    plt.title("Cropped Image")
    plt.imshow(cropped_rgb)
    plt.axis('off')
else:
    print("No circle detected.")

# Show final detection result
circle_detected_rgb = cv2.cvtColor(circle_detected, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6, 6))
plt.title("Circle Detected After Dark Blue Thresholding")
plt.imshow(circle_detected_rgb)
plt.axis('off')
plt.tight_layout()
plt.show()
