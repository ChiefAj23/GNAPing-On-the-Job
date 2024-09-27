import cv2
import numpy as np
import os
import torch
from cv2.ximgproc import guidedFilter

"""
This script applies image processing techniques such as combined filtering and Laplacian pyramid 
transformations to simulate attacks on images. It also computes noise levels from these attacks, 
saves both the attacked images and noise maps, and processes images from a dataset folder.

Functions:
1. compute_noise_s: Computes the difference (noise) between original and filtered images.
2. pad_image_to_even_dimensions: Pads the image to ensure its dimensions are even.
3. build_laplacian_pyramid: Constructs a Laplacian pyramid for multi-scale image representation.
4. reconstruct_from_laplacian_pyramid: Reconstructs the image from the Laplacian pyramid.
5. apply_combined_filter: Applies guided and Laplacian filtering to the image, returning the noisy image.
6. apply_laplacian_pyramid: Applies a Laplacian pyramid transformation to the image.
7. process_images: Processes all images in a folder by applying various attacks and saving results.

TODO:
1. Add argument parsing for noise scale and file paths.
2. Improve exception handling for corrupted or incorrectly sized images.
"""

# Load YOLOv5 model 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  

# Compute the pixel-wise noise difference between the original and filtered image.
def compute_noise_s(image: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) - filtered.astype(np.float32)

# Pad the image to ensure its dimensions are even.
def pad_image_to_even_dimensions(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    padded_image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

# Build a Laplacian pyramid by repeatedly downsampling and subtracting the upsampled image.
def build_laplacian_pyramid(image: np.ndarray, levels: int) -> list:
    pyramid = []
    current_image = image
    for _ in range(levels):
        current_image = pad_image_to_even_dimensions(current_image)
        down = cv2.pyrDown(current_image)
        up = cv2.pyrUp(down, dstsize=(current_image.shape[1], current_image.shape[0]))
        laplacian = cv2.subtract(current_image, up)
        pyramid.append(laplacian)
        current_image = down
    pyramid.append(current_image) 
    return pyramid

# Reconstruct the image from its Laplacian pyramid by progressively adding the upsampled layers.
def reconstruct_from_laplacian_pyramid(pyramid: list) -> np.ndarray:
    image = pyramid[-1]  
    for layer in reversed(pyramid[:-1]):
        layer = pad_image_to_even_dimensions(layer)
        image = pad_image_to_even_dimensions(image)
        image = cv2.pyrUp(image, dstsize=(layer.shape[1], layer.shape[0]))
        image = cv2.add(image, layer)  
    return image

# Apply a combined guided and Laplacian filter to the image and return both the noisy image and noise map.
def apply_combined_filter(image: np.ndarray, sigma: float, noise_scale: float, noise_function) -> (np.ndarray, np.ndarray):
    guided_radius = 5
    guided_eps = 0.1
    guided_image = guidedFilter(image, image, guided_radius, guided_eps)
    guided_noise = noise_function(image, guided_image)
    blurred = cv2.GaussianBlur(guided_image, (0, 0), sigma)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log = cv2.convertScaleAbs(log)
    log_noise = noise_function(guided_image, log)
    combined_noise = guided_noise + log_noise
    scaled_noise = combined_noise * noise_scale
    noisy_image = cv2.subtract(image, cv2.convertScaleAbs(scaled_noise))
    return noisy_image, scaled_noise

# Apply the Laplacian pyramid method to the image and return both the reconstructed image and noise map.
def apply_laplacian_pyramid(image: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    try:
        image = pad_image_to_even_dimensions(image)
        pyramid = build_laplacian_pyramid(image, levels)
        reconstructed_image = reconstruct_from_laplacian_pyramid(pyramid)
        laplacian_noise = compute_noise_s(image, reconstructed_image)
        return reconstructed_image, laplacian_noise
    except cv2.error as e:
        print(f"Skipping image due to pyramid size mismatch: {e}")
        return None, None

# Processes images from the input folder by applying attacks (filters) and saving results to the output folder.
def process_images(input_folder: str, output_folder: str, noise_scale: float):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    processed_count = 0

    for filename in os.listdir(input_folder):
        if processed_count >= 5:  
            break

        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Ensure the image dimensions are even
            h, w = image.shape[:2]
            if h % 2 != 0 or w % 2 != 0:
                continue

            # Save the original clean image
            clean_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_clean{os.path.splitext(filename)[1]}")
            cv2.imwrite(clean_image_path, image)

            # Apply the combined filter attack
            combined_image, combined_noise = apply_combined_filter(image, sigma=1.0, noise_scale=noise_scale, noise_function=compute_noise_s)
            combined_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_combined{os.path.splitext(filename)[1]}")
            combined_noise_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_combined_noise{os.path.splitext(filename)[1]}")
            cv2.imwrite(combined_image_path, combined_image)
            cv2.imwrite(combined_noise_path, combined_noise)

            # Apply the Laplacian pyramid attack
            laplacian_image, laplacian_noise = apply_laplacian_pyramid(image, levels=3)
            if laplacian_image is None:
                os.remove(clean_image_path)
                os.remove(combined_image_path)
                os.remove(combined_noise_path)
                continue 

            laplacian_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_laplacian{os.path.splitext(filename)[1]}")
            laplacian_noise_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_laplacian_noise{os.path.splitext(filename)[1]}")
            cv2.imwrite(laplacian_image_path, laplacian_image)
            cv2.imwrite(laplacian_noise_path, laplacian_noise)

            # Apply both attacks together
            both_attacks_image, both_combined_noise = apply_combined_filter(laplacian_image, sigma=1.0, noise_scale=noise_scale, noise_function=compute_noise_s)
            both_noise = both_combined_noise + laplacian_noise
            both_attacks_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_both{os.path.splitext(filename)[1]}")
            both_noise_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_both_noise{os.path.splitext(filename)[1]}")
            cv2.imwrite(both_attacks_image_path, both_attacks_image)
            cv2.imwrite(both_noise_path, both_noise)

            processed_count += 1
            print(f"Processed and saved: {filename}")

    print(f"Processing complete. {processed_count} images processed.")

if __name__ == "__main__":
    input_folder = r"C:\Users\Rttho\Downloads\train2017\train2017"  # Update this with the path to your dataset
    output_folder = 'processed_images_attacks_noise'
    noise_scale = 0.7

    process_images(input_folder, output_folder, noise_scale)
