import cv2
import numpy as np
import os
import csv
from cv2.ximgproc import guidedFilter

"""
This script loads a Caffe model and performs object detection on images, applies various 
filters including a combined guided and Laplacian filter, and processes images through 
different attack and defense mechanisms. The results are saved along with object detection 
metrics (e.g., confidence) in an output directory.

Functions:
1. load_caffe_model: Loads the Caffe model from specified paths.
2. detect_objects_caffe: Performs object detection on the image using the loaded Caffe model.
3. annotate_image_caffe: Annotates the image with bounding boxes and confidence levels.
4. resize_to_fixed_size: Resizes the image to a fixed size for consistency.
5. resize_to_original_size: Resizes the image back to its original size.
6. build_laplacian_pyramid: Constructs a Laplacian pyramid for multi-scale processing.
7. reconstruct_from_laplacian_pyramid: Reconstructs the image from its Laplacian pyramid.
8. apply_combined_filter: Applies a guided filter and Laplacian filter to an image.
9. apply_laplacian_pyramid: Applies the Laplacian pyramid to enhance images.
10. compute_noise_s: Computes noise between original and filtered images.
11. pad_image_to_even_dimensions: Pads an image to ensure even dimensions.
12. apply_unsharp_mask: Enhances the image using unsharp mask filtering.
13. process_images: Processes a batch of images, applies attacks/defenses, and saves results.

TODO:
1. Add parameter parsing for command-line execution.
2. Improve exception handling for file and model loading errors.
"""

# Load the Caffe model from specified .prototxt and .caffemodel files.
def load_caffe_model(prototxt_path, model_path):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net

# Perform object detection on the image using the loaded Caffe model and return detections.
def detect_objects_caffe(image, net):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    return detections

# Annotate the image with bounding boxes and confidence levels for detected objects.
def annotate_image_caffe(image, detections, confidence_threshold=0.5):
    objects_info = []
    highest_conf = 0
    highest_conf_label = ""
    height, width = image.shape[:2]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            label = f"Person: {confidence:.2f}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            objects_info.append(("Person", confidence))
            if confidence > highest_conf:
                highest_conf = confidence
                highest_conf_label = "Person"

    mean_conf = np.mean([conf for _, conf in objects_info]) if objects_info else 0

    return image, objects_info, highest_conf, highest_conf_label, mean_conf

# Resize the image to a fixed size for consistent processing.
def resize_to_fixed_size(image: np.ndarray, size=(512, 512)) -> np.ndarray:
    return cv2.resize(image, size)

# Resize the image back to its original size after processing.
def resize_to_original_size(image: np.ndarray, original_size) -> np.ndarray:
    return cv2.resize(image, (original_size[1], original_size[0]))

# Build a Laplacian pyramid for multi-scale image processing.
def build_laplacian_pyramid(image: np.ndarray, levels: int) -> list:
    pyramid = []
    current_image = image
    for _ in range(levels):
        down = cv2.pyrDown(current_image)
        up = cv2.pyrUp(down, dstsize=(current_image.shape[1], current_image.shape[0]))
        laplacian = cv2.subtract(current_image, up)
        pyramid.append(laplacian)
        current_image = down
    pyramid.append(current_image)  
    return pyramid

# Reconstruct the original image from its Laplacian pyramid.
def reconstruct_from_laplacian_pyramid(pyramid: list) -> np.ndarray:
    image = pyramid[-1]  
    for layer in reversed(pyramid[:-1]):
        image = cv2.pyrUp(image, dstsize=(layer.shape[1], layer.shape[0]))
        image = cv2.add(image, layer)
    return image

# Apply a combined guided filter and Laplacian filter to enhance the image.
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

# Apply the Laplacian pyramid method to the image and return the enhanced version.
def apply_laplacian_pyramid(image: np.ndarray, levels: int) -> np.ndarray:
    try:
        original_size = image.shape[:2]
        resized_image = resize_to_fixed_size(image)
        pyramid = build_laplacian_pyramid(resized_image, levels)
        reconstructed_image = reconstruct_from_laplacian_pyramid(pyramid)
        final_image = resize_to_original_size(reconstructed_image, original_size)
        return final_image
    except cv2.error as e:
        print(f"Skipping image due to pyramid size mismatch: {e}")
        return None

# Compute the noise between the original and filtered images by calculating the pixel-wise difference.
def compute_noise_s(image: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) - filtered.astype(np.float32)

# Pad the image with black pixels if necessary to ensure even dimensions.
def pad_image_to_even_dimensions(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    padded_image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

# Apply an unsharp mask to the image for enhancement.
def apply_unsharp_mask(image: np.ndarray, sigma=1.0, strength=20) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

# Process all images in the input folder by applying attacks and defenses, and save the results.
def process_images(input_folder: str, output_folder: str, noise_scale: float, prototxt_path: str, model_path: str):
    net = load_caffe_model(prototxt_path, model_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    csv_file = os.path.join(output_folder, "confidence_levels.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Type', 'Detected Objects', 'Number of Objects', 'Highest Confidence Object', 'Highest Confidence', 'Mean Confidence'])

        processed_count = 0

        for root, _, files in os.walk(input_folder):
            for filename in files:
                if processed_count >= 100:
                    break

                if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(root, filename)
                    image = cv2.imread(image_path)

                    # Process clean image
                    detections = detect_objects_caffe(image, net)
                    clean_image, clean_objects_info, clean_highest_conf, clean_highest_conf_label, clean_mean_conf = annotate_image_caffe(image.copy(), detections)
                    clean_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_clean{os.path.splitext(filename)[1]}")
                    cv2.imwrite(clean_image_path, clean_image)
                    clean_confidence_data = [filename, 'Clean', clean_objects_info, len(clean_objects_info), clean_highest_conf_label, clean_highest_conf, clean_mean_conf]

                    # Apply attack (Combined Filter)
                    combined_attacked_image = apply_combined_filter(image, sigma=1.0, noise_scale=noise_scale, noise_function=compute_noise_s)
                    combined_attacked_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_LoG_attack{os.path.splitext(filename)[1]}")
                    cv2.imwrite(combined_attacked_image_path, combined_attacked_image)
                    detections = detect_objects_caffe(combined_attacked_image, net)
                    combined_attacked_image_annotated, combined_attacked_objects_info, combined_attacked_highest_conf, combined_attacked_highest_conf_label, combined_attacked_mean_conf = annotate_image_caffe(combined_attacked_image.copy(), detections)
                    combined_attacked_confidence_data = [filename, 'LoG_attack', combined_attacked_objects_info, len(combined_attacked_objects_info), combined_attacked_highest_conf_label, combined_attacked_highest_conf, combined_attacked_mean_conf]

                    # Apply attack (Laplacian Pyramid)
                    laplacian_attacked_image = apply_laplacian_pyramid(image, levels=3)
                    if laplacian_attacked_image is None:
                        os.remove(clean_image_path)
                        os.remove(combined_attacked_image_path)
                        continue  

                    # Process laplacian attacked image
                    detections = detect_objects_caffe(laplacian_attacked_image, net)
                    laplacian_attacked_image_annotated, laplacian_attacked_objects_info, laplacian_attacked_highest_conf, laplacian_attacked_highest_conf_label, laplacian_attacked_mean_conf = annotate_image_caffe(laplacian_attacked_image.copy(), detections)
                    laplacian_attacked_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_Pyramid_attack{os.path.splitext(filename)[1]}")
                    cv2.imwrite(laplacian_attacked_image_path, laplacian_attacked_image_annotated)
                    laplacian_attacked_confidence_data = [filename, 'Pyramid_attack', laplacian_attacked_objects_info, len(laplacian_attacked_objects_info), laplacian_attacked_highest_conf_label, laplacian_attacked_highest_conf, laplacian_attacked_mean_conf]

                    # Apply attack (Combined + Laplacian Pyramid)
                    combined_laplacian_attacked_image = apply_combined_filter(image, sigma=1.0, noise_scale=noise_scale, noise_function=compute_noise_s)
                    combined_laplacian_attacked_image = apply_laplacian_pyramid(combined_laplacian_attacked_image, levels=3)
                    if combined_laplacian_attacked_image is None:
                        os.remove(clean_image_path)
                        os.remove(combined_attacked_image_path)
                        os.remove(laplacian_attacked_image_path)
                        continue  
                    # Process combined + laplacian attacked image
                    detections = detect_objects_caffe(combined_laplacian_attacked_image, net)
                    combined_laplacian_attacked_image_annotated, combined_laplacian_attacked_objects_info, combined_laplacian_attacked_highest_conf, combined_laplacian_attacked_highest_conf_label, combined_laplacian_attacked_mean_conf = annotate_image_caffe(combined_laplacian_attacked_image.copy(), detections)
                    combined_laplacian_attacked_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_GNAAP_attack{os.path.splitext(filename)[1]}")
                    cv2.imwrite(combined_laplacian_attacked_image_path, combined_laplacian_attacked_image_annotated)
                    combined_laplacian_attacked_confidence_data = [filename, 'Combined_Laplacian_Attacked', combined_laplacian_attacked_objects_info, len(combined_laplacian_attacked_objects_info), combined_laplacian_attacked_highest_conf_label, combined_laplacian_attacked_highest_conf, combined_laplacian_attacked_mean_conf]

                    # Apply defense (Unsharp Mask)
                    defended_image = apply_unsharp_mask(combined_laplacian_attacked_image, sigma=1.0, strength=14)
                    results = detect_objects_caffe(defended_image, net)
                    defended_image_annotated, defended_objects_info, defended_highest_conf, defended_highest_conf_label, defended_mean_conf = annotate_image_caffe(defended_image.copy(), results)
                    defended_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_defended{os.path.splitext(filename)[1]}")
                    cv2.imwrite(defended_image_path, defended_image_annotated)
                    defended_confidence_data = [filename, 'Defended', defended_objects_info, len(defended_objects_info), defended_highest_conf_label, defended_highest_conf, defended_mean_conf]

                    # Save data to CSV only if all versions are processed
                    writer.writerow(clean_confidence_data)
                    writer.writerow(combined_attacked_confidence_data)
                    writer.writerow(laplacian_attacked_confidence_data)
                    writer.writerow(combined_laplacian_attacked_confidence_data)
                    writer.writerow(defended_confidence_data)

                    processed_count += 1
                    print(f"Processed and saved: {filename}")

        print(f"Processing complete. {processed_count} images processed.")

if __name__ == "__main__":
    input_folder = r"C:\Users\Rttho\Downloads\lfw-a\lfw"  # Update this with path to your parent folder
    output_folder = 'processed_image_lfw'
    noise_scale = 0.7
    prototxt_path = r"C:\Users\Rttho\OneDrive - Lander University\Desktop\ComputerVision\models\deploy.prototxt.txt"  #Update this with the path to your deploy.prototxt file
    model_path = r"C:\Users\Rttho\OneDrive - Lander University\Desktop\ComputerVision\models\res10_300x300_ssd_iter_140000.caffemodel"  #Update this with the path to your caffemodel file

    process_images(input_folder, output_folder, noise_scale, prototxt_path, model_path)
