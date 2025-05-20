import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
# Start video capture (0 = default camera; change to 1 or 2 for other USB cameras)
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot access camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame for consistent processing (optional)
    frame = cv2.resize(frame, (640, 480))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold for dark blue to nearly black
    lower_dark_blue = np.array([100, 0, 0])
    upper_dark_blue = np.array([150, 255, 120])
    mask_dark_blue = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)

    # Bitwise AND to isolate dark blue areas
    result_dark_blue = cv2.bitwise_and(frame, frame, mask=mask_dark_blue)

    # Convert to grayscale for HoughCircles
    gray_masked = cv2.cvtColor(result_dark_blue, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_masked, 10, 255, cv2.THRESH_BINARY)
    blurred_mask = cv2.GaussianBlur(binary_mask, (9, 9), 2)
    # Detect circles using Hough Transform
    circles_masked = cv2.HoughCircles(
        blurred_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=120,
        maxRadius=300
    )

    # Draw circles
    output = frame.copy()
    if circles_masked is not None:
        circles_masked = np.uint16(np.around(circles_masked))
           # Skip frames where only one circle is detected
        print(len(circles_masked))
        largest_circle = max(circles_masked[0, :], key=lambda c: c[2])

        x, y, r = largest_circle
        if(r<150):
            continue
        cv2.circle(output, (x, y), r, (0, 0, 255), 3)
        cv2.circle(output, (x, y), 2, (0, 255, 0), 3)

    # Show original and mask
    cv2.imshow("Camera Feed", output)
    cv2.imshow("Binary Mask", blurred_mask)
    time.sleep(0.5)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
