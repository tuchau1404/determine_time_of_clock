import cv2
import datetime
import os

# Create 'images' folder if it doesn't exist
output_folder = 'images/right_angle'
os.makedirs(output_folder, exist_ok=True)

# Open the default camera (0 = first camera device)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("Press 'J' to capture an image. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('j') or key == ord('J'):
        # Use timestamp for unique filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_folder, f'capture_{timestamp}.png')
        cv2.imwrite(filename, frame)
        print(f'Image saved as {filename}')
    elif key == ord('q') or key == ord('Q'):
        print("Quitting...")
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()