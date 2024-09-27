import cv2
import numpy as np
import os

"""
This script processes an image by applying an unsharp mask as a defense method. The processed (defended) image is saved to a specified output directory.

Functions:
1. apply_unsharp_mask: Applies unsharp masking to the image to enhance sharpness as a defense technique.
2. process_and_save_image: Reads an image, applies the unsharp mask, and saves the result to an output directory.

TODO:
1. Add support for batch processing of multiple images in a folder.
2. Implement different defense techniques for comparison.
"""

# Applies an unsharp mask to the image to enhance sharpness and reduce noise.
def apply_unsharp_mask(image: np.ndarray, sigma=1.0, strength=11, threshold=0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

# Reads the image, applies the defense method (unsharp mask), and saves the processed image to an output directory.
def process_and_save_image(image_path: str):
    image = cv2.imread(image_path)
    # Apply the unsharp mask as the defense method to enhance the image.
    defended_image = apply_unsharp_mask(image)
    
    # Create the output directory if it doesn't exist.
    output_dir = "defended_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Construct the output filename by appending "_defense" to the original name.
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}_defense{ext}"
    output_path = os.path.join(output_dir, new_name)
    
    # Save the defended image in the output directory.
    cv2.imwrite(output_path, defended_image)
    print(f"Defended image saved as {output_path}")

# Example usage: Reads a sample image, processes it, and saves the result.
if __name__ == "__main__":
    input_image_path = r"C:\Users\Rttho\OneDrive - Lander University\Desktop\ComputerVision\attacked_data_set\000000000009.jpg"  # Replace with the path to your image.
    process_and_save_image(input_image_path)

