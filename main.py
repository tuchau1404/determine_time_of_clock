from clock_reader import *

image_path = "images\\right_angle\\1.png"
image, binary = load_and_preprocess(image_path)


plt.figure(figsize=(4, 4))
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

circle = detect_circle(image, binary)
print(circle)
if not circle:
    print("No circle detected.")
else:
    cropped = crop_and_resize(image, circle)
    masked, mask = mask_circle(cropped)
    combined = apply_white_background(masked, mask)
    binary_mask = binary_threshold(combined)
    combined[binary_mask == 255] = 255  # Apply white mask

    red_mask, black_mask = apply_color_mask(combined)
    center = (100, 100)

    contours = get_contour_centers(black_mask)
    hour, minute, second, points = detect_time(center, contours, red_mask)
    visualize_time(black_mask, points, center, hour, minute, second)
