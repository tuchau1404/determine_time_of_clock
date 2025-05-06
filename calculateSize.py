import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# === Step 1: Load and preprocess image ===
image_path = "images\\Figure_9.png"
image = Image.open(image_path).convert("L")
image_np = np.array(image)
_, binary = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === Step 2: Get center of the image ===
height, width = image_np.shape
center_x, center_y = width // 2, height // 2
center = np.array([center_x, center_y])

# === Step 3: Sort contours by area (larger = hour hand, smaller = minute hand) ===
areas_and_contours = sorted([(cv2.contourArea(cnt), cnt) for cnt in contours], reverse=True)

labels = ["minute", "hour"]
angles = {}
points = {}

# === Step 4: Calculate angles from center to each hand ===
for label, (area, cnt) in zip(labels, areas_and_contours):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    vector = np.array([cx - center_x, cy - center_y])
    angle_rad = np.arctan2(-vector[1], vector[0])  # Flip y-axis for image coords
    angle_deg = (90 - np.degrees(angle_rad)) % 360
    angles[label] = angle_deg
    points[label] = (cx, cy)

# === Step 5: Calculate time ===
minute_angle = angles["minute"]
hour_angle = angles["hour"]

minute = round(minute_angle / 6)  # 6° per minute
hour_fraction = hour_angle / 30   # 30° per hour
hour = int(hour_fraction)

if minute == 60:
    minute = 0
    hour = (hour + 1) % 12

# === Step 6: Visualize result ===
image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
cv2.circle(image_rgb, (center_x, center_y), 4, (0, 0, 255), -1)  # Red center
cv2.circle(image_rgb, points["minute"], 5, (255, 0, 0), -1)       # Blue = minute
cv2.circle(image_rgb, points["hour"], 5, (0, 255, 255), -1)       # Yellow = hour
cv2.line(image_rgb, (center_x, center_y), points["minute"], (255, 0, 0), 2)
cv2.line(image_rgb, (center_x, center_y), points["hour"], (0, 255, 255), 2)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
plt.title(f"Calculated Time: {hour:02}:{minute:02}")
plt.axis("off")
plt.show()

# Output time
print(f"Time is approximately: {hour:02}:{minute:02}")
