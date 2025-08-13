import cv2
import os
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(gray)
    return clahe_image

def apply_median_filter(image, ksize=3):
    return cv2.medianBlur(image, ksize)

def apply_unsharp_masking(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    low_contrast_mask = np.absolute(image - blurred) < threshold
    np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def process_image(image):
    clahe_image = apply_clahe(image)
    median_filtered_image = apply_median_filter(clahe_image)
    sharpened_image = apply_unsharp_masking(median_filtered_image)
    return sharpened_image

def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Image {filename} could not be loaded.")
                continue

            processed_image = process_image(image)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)
            print(f"Processed {filename} and saved to {output_path}")

input_folder = r'your input path'
output_folder = r'your output path'

process_images_in_folder(input_folder, output_folder)
