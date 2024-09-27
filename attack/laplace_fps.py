import cv2
import numpy as np
from cv2.ximgproc import guidedFilter
import time
import csv
from matplotlib import pyplot as plt

"""
This script captures real-time video feed, applies image processing filters, and computes noise based on different methods.
It also detects faces using a neural network and annotates the output. The results, including detection confidence, 
are saved in a CSV file for analysis.

Functions:
1. compute_noise_s: Computes the pixel-wise difference (noise) between the original and filtered image.
2. compute_noise_p: Computes the sum of the original and filtered image (CiPer-p logic).
3. ensure_even_dimensions: Ensures that image dimensions are even for consistent image processing.
4. build_laplacian_pyramid: Constructs a Laplacian pyramid for multi-scale image representation.
5. reconstruct_from_laplacian_pyramid: Reconstructs an image from its Laplacian pyramid.
6. apply_combined_filter: Applies a guided filter combined with Laplacian filtering to an image.
7. apply_laplacian_pyramid: Applies the Laplacian pyramid method to enhance the image.
8. apply_unsharp_mask: Applies unsharp masking for image sharpening.
9. detect_faces_and_calculate_confidence: Detects faces and calculates the average confidence of detected faces.
10. annotate_image: Annotates the image with bounding boxes and confidence scores for detected faces.
11. save_to_csv: Saves frame data and settings to a CSV file.
12. main: Captures video frames, applies filters, toggles settings, and displays the annotated image with performance data.

TODO:
1. Improve video frame processing speed for real-time applications.
2. Add argument parsing for flexibility in setting parameters like noise scale and strength.
"""

# Computes the noise as the difference between the original and filtered image.
def compute_noise_s(image: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) - filtered.astype(np.float32)

# Computes the noise as the sum of the original and filtered image (CiPer-p logic).
def compute_noise_p(image: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) + filtered.astype(np.float32)

# Ensures the dimensions of the image are even, which is required for pyramid processing.
def ensure_even_dimensions(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return image[:new_h, :new_w]

# Builds a Laplacian pyramid for multi-scale image representation, capturing details at various levels.
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

# Reconstructs the original image by upsampling and adding layers of the Laplacian pyramid.
def reconstruct_from_laplacian_pyramid(pyramid: list) -> np.ndarray:
    image = pyramid[-1]  # Start with the smallest (base) image in the pyramid.
    for layer in reversed(pyramid[:-1]):
        image = cv2.pyrUp(image, dstsize=(layer.shape[1], layer.shape[0]))
        image = cv2.add(image, layer)
    return image

# Applies a combined guided filter and Laplacian filter to an image, enhancing details and reducing noise.
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

# Applies the Laplacian pyramid to the image for multi-scale processing and reconstruction.
def apply_laplacian_pyramid(image: np.ndarray, levels: int) -> np.ndarray:
    image = ensure_even_dimensions(image)
    pyramid = build_laplacian_pyramid(image, levels)
    reconstructed_image = reconstruct_from_laplacian_pyramid(pyramid)
    return reconstructed_image

# Applies an unsharp mask to the image for sharpening by enhancing high-frequency details.
def apply_unsharp_mask(image: np.ndarray, sigma=1.0, strength=1.5, threshold=0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

# Detects faces in the image using a neural network and calculates the average confidence of detections.
def detect_faces_and_calculate_confidence(image, net):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    avg_confidence = sum(detections[0, 0, i, 2] for i in range(detections.shape[2]) if detections[0, 0, i, 2] > 0.5) / (np.sum(detections[0, 0, :, 2] > 0.5) or 1)
    return avg_confidence, detections

# Annotates detected faces in the image with bounding boxes and confidence levels.
def annotate_image(image, detections):
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, f"{confidence:.2f}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return image

# Saves processed frame data and settings to a CSV file for later analysis.
def save_to_csv(csv_writer, data):
    csv_writer.writerow(data)

# Main function to capture video frames, apply filters, detect faces, and toggle settings.
def main():
    prototxt_path = r'C:\Users\Rttho\OneDrive - Lander University\Desktop\ComputerVision\models\deploy.prototxt.txt'
    model_path = r'C:\Users\Rttho\OneDrive - Lander University\Desktop\ComputerVision\models\res10_300x300_ssd_iter_140000.caffemodel'
    
    sigma = 0.1
    noise_scale = 0.6

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    vs = cv2.VideoCapture(0)  

    frame_count = 0
    levels = 3
    strength = 8
    filter_on = True
    pyramid_on = True
    mask_on = False  
    noise_method = compute_noise_s  

    csv_file_path = "output_settings.csv"
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Frame", "Filter On", "Pyramid On", "Low Pass On", "Noise Method", "Log Noise Scale", "Strength", "Confidence"])

        while True:
            start_time = time.time()
            ret, frame = vs.read()
            if not ret:
                break

            if frame_count % 2 == 0:
                frame = cv2.resize(frame, (320, 240))

                # Apply filters based on the toggles.
                if mask_on:
                    frame = apply_unsharp_mask(frame, strength=strength)
                if filter_on and pyramid_on:
                    filtered_frame = apply_combined_filter(frame, sigma, noise_scale, noise_method)
                    filtered_frame = apply_laplacian_pyramid(filtered_frame, levels=levels)
                elif filter_on:
                    filtered_frame = apply_combined_filter(frame, sigma, noise_scale, noise_method)
                elif pyramid_on:
                    filtered_frame = apply_laplacian_pyramid(frame, levels=3)
                else:
                    filtered_frame = frame

                avg_confidence_filtered, detections_filtered = detect_faces_and_calculate_confidence(filtered_frame, net)

                annotated_filtered = annotate_image(filtered_frame.copy(), detections_filtered)

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                cv2.putText(annotated_filtered, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(annotated_filtered, f"Noise Scale: {noise_scale:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow('Attacked Image', annotated_filtered)

            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                filter_on = not filter_on
                save_to_csv(csvwriter, [frame_count, filter_on, pyramid_on, mask_on, "CiPer-p" if noise_method == compute_noise_p else "Subtractive", noise_scale, strength, avg_confidence_filtered])
                print(f"Filter toggled to {'on' if filter_on else 'off'}")
            elif key == ord('p'):
                pyramid_on = not pyramid_on
                save_to_csv(csvwriter, [frame_count, filter_on, pyramid_on, mask_on, "CiPer-p" if noise_method == compute_noise_p else "Subtractive", noise_scale, strength, avg_confidence_filtered])
                print(f"Laplacian pyramid toggled to {'on' if pyramid_on else 'off'}")
            elif key == ord('n'):
                noise_method = compute_noise_p if noise_method == compute_noise_s else compute_noise_s
                save_to_csv(csvwriter, [frame_count, filter_on, pyramid_on, mask_on, "CiPer-p" if noise_method == compute_noise_p else "Subtractive", noise_scale, strength, avg_confidence_filtered])
                print(f"Noise method toggled to {'CiPer-p' if noise_method == compute_noise_p else 'Subtractive'}")
            elif key == ord('i'):
                noise_scale += 0.05
                save_to_csv(csvwriter, [frame_count, filter_on, pyramid_on, mask_on, "CiPer-p" if noise_method == compute_noise_p else "Subtractive", noise_scale, strength, avg_confidence_filtered])
                print(f"Noise scale increased to {noise_scale:.2f}")
            elif key == ord('d'):
                noise_scale = max(0, noise_scale - 0.05)
                save_to_csv(csvwriter, [frame_count, filter_on, pyramid_on, mask_on, "CiPer-p" if noise_method == compute_noise_p else "Subtractive", noise_scale, strength, avg_confidence_filtered])
                print(f"Noise scale decreased to {noise_scale:.2f}")
            elif key == ord('m'):
                mask_on = not mask_on
                save_to_csv(csvwriter, [frame_count, filter_on, pyramid_on, mask_on, "CiPer-p" if noise_method == compute_noise_p else "Subtractive", noise_scale, strength, avg_confidence_filtered])
                print(f"Mask filter toggled to {'on' if mask_on else 'off'}")
            elif key == ord('s'):
                strength += 0.5
                save_to_csv(csvwriter, [frame_count, filter_on, pyramid_on, mask_on, "CiPer-p" if noise_method == compute_noise_p else "Subtractive", noise_scale, strength, avg_confidence_filtered])
                print(f"Strength increased to {strength:.1f}")
            elif key == ord('r'):
                strength = max(0, strength - 0.5)
                save_to_csv(csvwriter, [frame_count, filter_on, pyramid_on, mask_on, "CiPer-p" if noise_method == compute_noise_p else "Subtractive", noise_scale, strength, avg_confidence_filtered])
                print(f"Strength decreased to {strength:.1f}")
            elif key == ord('z'):
                frame_count = 0
                strength = 1.5
                filter_on = True
                pyramid_on = True
                mask_on = False  
                noise_method = compute_noise_s  
                save_to_csv(csvwriter, [frame_count, filter_on, pyramid_on, mask_on, "Subtractive", noise_scale, strength, avg_confidence_filtered])
                print("Settings reset to default")

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
