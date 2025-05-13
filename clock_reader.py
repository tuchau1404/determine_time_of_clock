import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 0, 0])
    upper = np.array([150, 255, 120])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return image, binary

def detect_circle(image, binary_mask):
    circles = cv2.HoughCircles(binary_mask, cv2.HOUGH_GRADIENT, 1.2, 100,
                                param1=100, param2=30, minRadius=50, maxRadius=300)
    if circles is None:
        return None
    circle = max(np.uint16(np.around(circles))[0], key=lambda c: c[2])
    return tuple(circle)

def crop_and_resize(image, circle, size=(200, 200)):
    x, y, r = circle
    x1, y1 = max(x - r, 0), max(y - r, 0)
    x2, y2 = min(x + r, image.shape[1]), min(y + r, image.shape[0])
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)

def mask_circle(image, radius=50):
    size = image.shape[0:2]
    center = (size[1] // 2, size[0] // 2)
    mask = np.zeros(size, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    circle_img = cv2.bitwise_and(image, image, mask=mask)
    return circle_img, mask

def apply_white_background(foreground, mask):
    white_bg = np.ones_like(foreground, dtype=np.uint8) * 255
    inverted = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(white_bg, white_bg, mask=inverted)
    return cv2.add(foreground, bg)

def binary_threshold(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary

def apply_color_mask(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)

    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))

    cv2.circle(red_mask, (100, 100), 20, 0, -1)
    cv2.circle(black_mask, (100, 100), 20, 0, -1)
    return red_mask, black_mask

def get_contour_centers(mask):
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_angle_from_center(center, point):
    vector = np.array(point) - np.array(center)
    angle_rad = np.arctan2(-vector[1], vector[0])
    angle_deg = (90 - np.degrees(angle_rad)) % 360
    return angle_deg

def detect_time(center, contours, red_mask):
    areas = sorted([(cv2.contourArea(c), c) for c in contours], reverse=True)
    angles = {}
    points = {}
    hour = minute = second = 0

    if len(areas) == 1:
        M = cv2.moments(areas[0][1])
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            angle = get_angle_from_center(center, (cx, cy))
            minute = round(angle / 6)
            hour = roung(angle //30)
            points["both"] = (cx, cy)
    elif len(areas) >= 2:
        for label, (_, cnt) in zip(["hour", "minute"], areas):
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            angle = get_angle_from_center(center, (cx, cy))
            angles[label] = angle
            points[label] = (cx, cy)

        minute = round(angles["minute"] / 6)
        hour = round(angles["hour"] // 30)

    # Process second hand
    contours_red = get_contour_centers(red_mask)
    if contours_red:
        largest = max(contours_red, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            angle = get_angle_from_center(center, (cx, cy))
            second = round(angle / 6) % 60
            points["second"] = (cx, cy)

    return hour % 12, minute % 60, second % 60, points

def visualize_time(mask, points, center, hour, minute, second):
    canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.circle(canvas, center, 4, (0, 0, 255), -1)

    if "minute" in points:
        cv2.circle(canvas, points["minute"], 5, (255, 0, 0), -1)
        cv2.line(canvas, center, points["minute"], (255, 0, 0), 2)
    if "hour" in points:
        cv2.circle(canvas, points["hour"], 5, (0, 255, 255), -1)
        cv2.line(canvas, center, points["hour"], (0, 255, 255), 2)
    if "second" in points:
        cv2.circle(canvas, points["second"], 5, (0, 0, 255), -1)
        cv2.line(canvas, center, points["second"], (0, 0, 255), 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"Time: {hour:02}:{minute:02}:{second:02}")
    plt.axis("off")
    plt.show()
