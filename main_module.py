# Re-import required libraries after reset
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_threshold(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_dark_blue = np.array([90, 0, 0])
    upper_dark_blue = np.array([150, 255, 120])
    mask = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)
    result = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return image, binary

def detect_and_crop_circle(image, binary_mask):
    circles = cv2.HoughCircles(binary_mask, cv2.HOUGH_GRADIENT, 1.2, 100,
                                param1=100, param2=30, minRadius=50, maxRadius=300)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = max(circles[0, :], key=lambda c: c[2])
        x1, y1 = max(x - r, 0), max(y - r, 0)
        x2, y2 = min(x + r, image.shape[1]), min(y + r, image.shape[0])
        cropped = image[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (200, 200), interpolation=cv2.INTER_AREA)
        return resized
    return None

def mask_circle_and_prepare_white_bg(resized):
    mask = np.ones((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)
    masked = cv2.bitwise_and(resized, resized, mask=mask)
    white_bg = np.ones_like(resized, dtype=np.uint8) * 255
    inverted_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(white_bg, white_bg, mask=inverted_mask)
    final = cv2.add(masked, background)
    return final

def isolate_color_regions(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 70, 50]), np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 50])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    cv2.circle(red_mask, (100, 100), 20, 0, -1)
    cv2.circle(black_mask, (100, 100), 20, 0, -1)
    return red_mask, black_mask

def calculate_clock_time(black_mask):
    image_np = np.array(black_mask)
    _, binary = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image_np.shape
    center = np.array([width // 2, height // 2])
    areas_and_contours = sorted([(cv2.contourArea(cnt), cnt) for cnt in contours], reverse=True)
    labels = ["hour", "minute"]
    angles, points = {}, {}
    for label, (area, cnt) in zip(labels, areas_and_contours):
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        vector = np.array([cx - center[0], cy - center[1]])
        angle_rad = np.arctan2(-vector[1], vector[0])
        angle_deg = (90 - np.degrees(angle_rad)) % 360
        angles[label] = angle_deg
        points[label] = (cx, cy)

    minute = round(angles["minute"] / 6)
    hour_fraction = angles["hour"] / 30
    hour = round(hour_fraction)
    if minute == 60:
        minute = 0
        hour = (hour + 1) % 12

    return hour, minute, center, points

def visualize_clock(image_gray, center, points, hour, minute):
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(image_rgb, tuple(center), 4, (0, 0, 255), -1)
    cv2.circle(image_rgb, points["minute"], 5, (255, 0, 0), -1)
    cv2.circle(image_rgb, points["hour"], 5, (0, 255, 255), -1)
    cv2.line(image_rgb, tuple(center), points["minute"], (255, 0, 0), 2)
    cv2.line(image_rgb, tuple(center), points["hour"], (0, 255, 255), 2)
    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
    plt.title(f"Calculated Time: {hour:02}:{minute:02}")
    plt.axis("off")
    plt.show()

def process_clock_image(image_path):
    image, binary_mask = load_and_threshold(image_path)
    resized = detect_and_crop_circle(image, binary_mask)
    if resized is None:
        print("No circle detected.")
        return
    final = mask_circle_and_prepare_white_bg(resized)
    grayscale = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    _, binary_thresh = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    thresholded_3ch = cv2.merge([binary_thresh] * 3)
    combined = final.copy()
    combined[thresholded_3ch == 255] = 255
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    _, black_mask = isolate_color_regions(combined_rgb)
    hour, minute, center, points = calculate_clock_time(black_mask)
    visualize_clock(black_mask, center, points, hour, minute)
    print(f"Time is approximately: {hour:02}:{minute:02}")

process_clock_image("images\\right_angle\\3.png")
