import cv2
import datetime
import os

# Create 'images' folder if it doesn't exist
output_folder = 'images/right_angle'
os.makedirs(output_folder, exist_ok=True)

# Try multiple camera indices to find one that works
cap = None
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)


print("Press 'J' to capture an image. Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('j') or key == ord('J'):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_folder, f'capture_{timestamp}.png')
        cv2.imwrite(filename, frame)
        print(f'Image saved as {filename}')
    elif key == ord('q') or key == ord('Q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
