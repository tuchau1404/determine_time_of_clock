import cv2
import numpy as np
import matplotlib.pyplot as plt


image_path = "images\\right_angle\\5.png"
image = cv2.imread(image_path)

# Convert to HSV for color thresholding
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV range for dark blue to nearly black
lower_dark_blue = np.array([90, 0, 0])
upper_dark_blue = np.array([150, 255, 120])

# Mask for dark blue range
mask_dark_blue = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)

# Apply mask to keep only dark blue parts
result_dark_blue = cv2.bitwise_and(image, image, mask=mask_dark_blue)

# Convert masked result to grayscale
gray_masked = cv2.cvtColor(result_dark_blue, cv2.COLOR_BGR2GRAY)

# Threshold to binary
_, binary_mask = cv2.threshold(gray_masked, 10, 255, cv2.THRESH_BINARY)
plt.figure(figsize=(4, 4))
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
    resized_cropped = cv2.resize(cropped_image, (200, 200), interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2RGB)
    # Show cropped image
    plt.figure(figsize=(4, 4))
    plt.title("Cropped Image")
    plt.imshow(cropped_rgb)
    plt.axis('off')
else:
    print("No circle detected.")

# Show final detection result
circle_detected_rgb = cv2.cvtColor(circle_detected, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(4, 4))
plt.title("Circle Detected After Dark Blue Thresholding")
plt.imshow(circle_detected_rgb)
plt.axis('off')

# Create a black background image (200x200 like the resized circle)
mask = np.ones((200, 200), dtype=np.uint8)

# Draw white filled circle at center (100, 100) with radius 50
cv2.circle(mask, (100, 100), 50, 255, -1)

# Apply the mask: keep the circle area, set the rest to black
masked_circle = cv2.bitwise_and(resized_cropped,resized_cropped, mask=mask)

# Display result
masked_rgb = cv2.cvtColor(masked_circle, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(4, 4))
plt.imshow(masked_rgb)
plt.title("Circle Mask at (100,100), R=50")
plt.axis("off")


# Create a white background
white_background = np.ones_like(resized_cropped, dtype=np.uint8) * 255

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
upper_black = np.array([180, 255, 80])

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


image_np = np.array(black_mask)
_, binary = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === Step 2: Get center of the image ===
height, width = image_np.shape
center_x, center_y = width // 2, height // 2
center = np.array([center_x, center_y])

# === Step 3: Sort contours by area (larger = hour hand, smaller = minute hand) ===
areas_and_contours = sorted([(cv2.contourArea(cnt), cnt) for cnt in contours], reverse=True)

angles = {}
points = {}

if len(areas_and_contours) == 1:
    # === Special Case: Hands are overlapping ===
    area, cnt = areas_and_contours[0]
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        vector = np.array([cx - center_x, cy - center_y])
        angle_rad = np.arctan2(-vector[1], vector[0])
        angle_deg = (90 - np.degrees(angle_rad)) % 360
        angles["both"] = angle_deg
        points["both"] = (cx, cy)

        # Assume hands overlap — derive time
        minute = round(angle_deg / 6)  # 6° per minute
        hour_fraction = angle_deg / 30
        hour = round(hour_fraction)

        if minute == 60:
            minute = 0
            hour = (hour + 1) % 12
    else:
        # Invalid moment
        hour = 0
        minute = 0
else:
    # === Normal Case: 2 contours for hour and minute hands ===
    labels = ["hour", "minute"]
    for label, (area, cnt) in zip(labels, areas_and_contours):
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        vector = np.array([cx - center_x, cy - center_y])
        angle_rad = np.arctan2(-vector[1], vector[0])  # Flip y-axis
        angle_deg = (90 - np.degrees(angle_rad)) % 360
        angles[label] = angle_deg
        points[label] = (cx, cy)

    minute_angle = angles["minute"]
    hour_angle = angles["hour"]

    minute = round(minute_angle / 6)
    hour_fraction = hour_angle / 30
    hour = round(hour_fraction)

    if minute == 60:
        minute = 0
        hour = (hour + 1) % 12

# === Process Red Mask (for Second Hand) ===
second_np = np.array(red_mask)
_, binary_red = cv2.threshold(second_np, 127, 255, cv2.THRESH_BINARY)
contours_red, _ = cv2.findContours(binary_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours_red:
    largest_red = max(contours_red, key=cv2.contourArea)
    M = cv2.moments(largest_red)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        vector = np.array([cx - center_x, cy - center_y])
        angle_rad = np.arctan2(-vector[1], vector[0])
        angle_deg = (90 - np.degrees(angle_rad)) % 360
        second = round(angle_deg / 6) % 60
        angles["second"] = angle_deg
        points["second"] = (cx, cy)
    else:
        second = 0
else:
    second = 0
# === Visualize result ===
image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
cv2.circle(image_rgb, (center_x, center_y), 4, (0, 0, 255), -1)  # Red center

if "minute" in points and "hour" in points:
    cv2.circle(image_rgb, points["minute"], 5, (255, 0, 0), -1)       # Blue = minute
    cv2.circle(image_rgb, points["hour"], 5, (0, 255, 255), -1)       # Yellow = hour
    cv2.line(image_rgb, (center_x, center_y), points["minute"], (255, 0, 0), 2)
    cv2.line(image_rgb, (center_x, center_y), points["hour"], (0, 255, 255), 2)
elif "both" in points:
    cv2.circle(image_rgb, points["both"], 5, (0, 255, 0), -1)          # Green = both
    cv2.line(image_rgb, (center_x, center_y), points["both"], (0, 255, 0), 2)

if "second" in points:
    cv2.circle(image_rgb, points["second"], 5, (0, 0, 255), -1)        # Red = second
    cv2.line(image_rgb, (center_x, center_y), points["second"], (0, 0, 255), 1)

# Show final result
plt.figure(figsize=(4, 4))
plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
plt.title(f"Calculated Time: {hour:02}:{minute:02}:{second:02}")
plt.axis("off")
plt.show()
