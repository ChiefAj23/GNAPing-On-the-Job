import cv2
import numpy as np
from cv2.ximgproc import guidedFilter
import os

"""
This script applies a combined guided and Laplacian filter to images, with the option to construct
and reconstruct images using a Laplacian pyramid. It processes all images in a specified folder
and saves the processed results to an output directory.

Functions:
1. compute_noise_s: Computes the pixel-wise noise between an original and filtered image.
2. ensure_even_dimensions: Ensures the image dimensions are even for pyramid operations.
3. build_laplacian_pyramid: Construct a Laplacian pyramid from an image.
4. reconstruct_from_laplacian_pyramid: Reconstruct an image from a Laplacian pyramid.
5. apply_combined_filter: Applies guided filtering and Laplacian filtering to enhance images.
6. apply_laplacian_pyramid: Applies the Laplacian pyramid method for multi-scale image processing.
7. process_images: Processes all images in a folder and saves the filtered results.

TODO:
1. Add parameter handling for custom noise scales and pyramid levels.
"""

# Compute the pixel-wise difference (noise) between the original and filtered images.
def compute_noise_s(image: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) - filtered.astype(np.float32)

# Ensure the image dimensions are even by trimming if necessary (required for pyramid operations).
def ensure_even_dimensions(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return image[:new_h, :new_w]

# Build a Laplacian pyramid for the image by progressively downsampling and calculating differences.
def build_laplacian_pyramid(image: np.ndarray, levels: int) -> list:
    pyramid = []
    current_image = image
    for _ in range(levels):
        current_image = ensure_even_dimensions(current_image)
        down = cv2.pyrDown(current_image)
        up = cv2.pyrUp(down, dstsize=(current_image.shape[1], current_image.shape[0]))
        laplacian = cv2.subtract(current_image, up)
        pyramid.append(laplacian)
        current_image = down
    pyramid.append(current_image)  
    return pyramid

# Reconstruct the original image by upsampling from the smallest image in the pyramid.
def reconstruct_from_laplacian_pyramid(pyramid: list) -> np.ndarray:
    image = pyramid[-1] 
    for layer in reversed(pyramid[:-1]):
        layer = ensure_even_dimensions(layer)
        image = ensure_even_dimensions(image)
        image = cv2.pyrUp(image, dstsize=(layer.shape[1], layer.shape[0]))
        image = cv2.add(image, layer)  
    return image

# Apply a combined guided filter and Laplacian filter to enhance the image and reduce noise.
def apply_combined_filter(image: np.ndarray, sigma: float, noise_scale: float, noise_function) -> np.ndarray:
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
    return noisy_image

# Apply the Laplacian pyramid method to the image, returning a multi-scale enhanced version.
def apply_laplacian_pyramid(image: np.ndarray, levels: int) -> np.ndarray:
    image = ensure_even_dimensions(image)
    pyramid = build_laplacian_pyramid(image, levels)
    reconstructed_image = reconstruct_from_laplacian_pyramid(pyramid)
    return reconstructed_image

# Process all images in the input folder by applying combined filtering and Laplacian pyramid processing.
# The processed images are saved in the specified output folder.
def process_images(input_folder: str, output_folder: str, noise_scale: float):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            try:
                filtered_image = apply_combined_filter(image, sigma=1.0, noise_scale=noise_scale, noise_function=compute_noise_s)
                attacked_image = apply_laplacian_pyramid(filtered_image, levels=3)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, attacked_image)
                print(f"Processed and saved: {output_path}")
            except cv2.error as e:
                print(f"Skipped {filename} due to size issue: {e}")

if __name__ == "__main__":
    input_folder = r"C:\Users\Rttho\Downloads\train2017\train2017" #update this with input code
    output_folder = 'attacked_data_set'
    noise_scale = 0.8  # Set the noise scaling factor for the image
    process_images(input_folder, output_folder, noise_scale)

