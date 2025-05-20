from clock_reader import *
config = load_config("config.yaml")
image, binary = load_and_preprocess(config['image_path'], config['color_thresholds']['dark_blue'])

if config['visualization']['show_original']:
    plt.figure(figsize=(4, 4))
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

circle = detect_circle(image, binary, config['circle_detection'])
if not circle:
    print("No circle detected.")
else:
    cropped = crop_and_resize(image, circle, config['crop']['resize_to'])
    masked, mask = mask_circle(cropped, config['mask']['radius'])
    combined = apply_white_background(masked, mask)
    binary_mask = binary_threshold(combined)
    combined[binary_mask == 255] = 255

    red_mask, black_mask = apply_color_mask(combined, config['color_thresholds'], config['mask']['center_padding'])
    center = (100, 100)

    contours = get_contour_centers(black_mask)
    hour, minute, second, points = detect_time(center, contours, red_mask)
    visualize_time(black_mask, points, center, hour, minute, second)