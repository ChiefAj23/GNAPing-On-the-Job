import cv2
import numpy as np
from cv2.ximgproc import guidedFilter

"""
This script applies various image processing techniques such as noise computation, unsharp masking, and 
Laplacian pyramid filtering to an image. It also includes real-time toggling of filters and noise settings, 
and detects faces using a neural network.

Functions:
1. compute_noise_s: Computes the noise as the difference between the original and filtered image.
2. compute_noise_p: Computes the noise as the sum of the original and filtered image (CiPer-p logic).
3. ensure_even_dimensions: Ensures the image dimensions are even for consistency in pyramid processing.
4. build_laplacian_pyramid: Constructs a Laplacian pyramid for multi-scale image representation.
5. reconstruct_from_laplacian_pyramid: Reconstructs an image from its Laplacian pyramid.
6. apply_combined_filter: Applies a combination of guided and Laplacian filters to the image.
7. apply_laplacian_pyramid: Applies a Laplacian pyramid transformation to enhance the image.
8. apply_unsharp_mask: Sharpens the image using unsharp masking for defense against noise.
9. detect_faces_and_calculate_confidence: Detects faces in the image and calculates the average confidence score.
10. annotate_image: Annotates detected faces with bounding boxes and confidence levels.
11. main: Main loop for applying filters and toggling settings in real-time.

"""

# Computes the noise as the difference between the original and filtered image.
def compute_noise_s(image: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) - filtered.astype(np.float32)

# Computes the noise as the sum of the original and filtered image (CiPer-p logic).
def compute_noise_p(image: np.ndarray, filtered: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) + filtered.astype(np.float32)

# Ensures the image dimensions are even, required for pyramid processing.
def ensure_even_dimensions(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    return image[:new_h, :new_w]

# Builds a Laplacian pyramid for multi-scale image representation.
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

# Reconstructs an image from its Laplacian pyramid by progressively adding layers.
def reconstruct_from_laplacian_pyramid(pyramid: list) -> np.ndarray:
    image = pyramid[-1]  
    for layer in reversed(pyramid[:-1]):
        image = cv2.pyrUp(image, dstsize=(layer.shape[1], layer.shape[0]))
        image = cv2.add(image, layer)  
    return image

# Applies a combination of guided and Laplacian filters to the image, enhancing details while reducing noise.
def apply_combined_filter(image: np.ndarray, log_sigma: float, log_noise_scale: float, noise_function) -> np.ndarray:
    guided_radius = 5
    guided_eps = 0.1
    guided_image = guidedFilter(image, image, guided_radius, guided_eps)
    guided_noise = noise_function(image, guided_image)
    blurred = cv2.GaussianBlur(guided_image, (0, 0), log_sigma)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log = cv2.convertScaleAbs(log)
    log_noise = noise_function(guided_image, log)
    combined_noise = guided_noise + log_noise
    scaled_noise = combined_noise * log_noise_scale
    noisy_image = cv2.subtract(image, cv2.convertScaleAbs(scaled_noise))
    return noisy_image

# Applies the Laplacian pyramid to the image for multi-scale enhancement.
def apply_laplacian_pyramid(image: np.ndarray, levels: int) -> np.ndarray:
    image = ensure_even_dimensions(image)
    pyramid = build_laplacian_pyramid(image, levels)
    reconstructed_image = reconstruct_from_laplacian_pyramid(pyramid)
    return reconstructed_image

# Applies unsharp masking to the image for sharpening by enhancing high-frequency details.
def apply_unsharp_mask(image: np.ndarray, sigma=1.0, strength=1.5, threshold=0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    
    return sharpened

# Detects faces in the image using a pre-trained neural network model and calculates the average confidence score.
def detect_faces_and_calculate_confidence(image, net):
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    avg_confidence = sum(detections[0, 0, i, 2] for i in range(detections.shape[2]) if detections[0, 0, i, 2] > 0.5) / (np.sum(detections[0, 0, :, 2] > 0.5) or 1)
    return avg_confidence, detections

# Annotates the image with bounding boxes and confidence levels for detected faces.
def annotate_image(image, detections):
    avg_confidence = 0
    count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            avg_confidence += confidence
            count += 1
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, f"{confidence:.2f}", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    if count > 0:
        avg_confidence /= count
    return image, avg_confidence

# Main function for real-time image filtering and face detection.
def main():
    prototxt_path = r'C:\Users\Rttho\OneDrive - Lander University\Desktop\ComputerVision\models\deploy.prototxt.txt'
    model_path = r'C:\Users\Rttho\OneDrive - Lander University\Desktop\ComputerVision\models\res10_300x300_ssd_iter_140000.caffemodel'
    
    log_sigma = 0.1
    log_noise_scale = 0.77

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    image_path = r"C:\Users\Rttho\OneDrive - Lander University\Pictures\Camera Roll\WIN_20240621_14_33_21_Pro.jpg"  # Change to your image path
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error loading image {image_path}")
        return

    original_image = cv2.resize(original_image, (320, 240))

    strength = 11.0
    filter_on = True
    pyramid_on = True
    low_pass_on = False 
    noise_method = compute_noise_s 

    while True:
        frame = original_image.copy()

        if low_pass_on:
            frame = apply_unsharp_mask(frame, strength=strength)
        if filter_on and pyramid_on:
            filtered_frame = apply_combined_filter(frame, log_sigma, log_noise_scale, noise_method)
            filtered_frame = apply_laplacian_pyramid(filtered_frame, levels=3)
        elif filter_on:
            filtered_frame = apply_combined_filter(frame, log_sigma, log_noise_scale, noise_method)
        elif pyramid_on:
            filtered_frame = apply_laplacian_pyramid(frame, levels=3)
        else:
            filtered_frame = frame

        annotated_filtered, avg_confidence_filtered = annotate_image(filtered_frame.copy(), detect_faces_and_calculate_confidence(filtered_frame, net)[1])


        cv2.putText(annotated_filtered, f"Noise Scale: {log_noise_scale:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated_filtered, f"Strength: {strength:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated_filtered, f"Confidence: {avg_confidence_filtered:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Attacked Image', annotated_filtered)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            filter_on = not filter_on
            print(f"Filter toggled to {'on' if filter_on else 'off'}")
        elif key == ord('p'):
            pyramid_on = not pyramid_on
            print(f"Laplacian pyramid toggled to {'on' if pyramid_on else 'off'}")
        elif key == ord('n'):
            noise_method = compute_noise_p if noise_method == compute_noise_s else compute_noise_s
            print(f"Noise method toggled to {'CiPer-p' if noise_method == compute_noise_p else 'Subtractive'}")
        elif key == ord('i'):
            log_noise_scale += 0.01
            print(f"Noise scale increased to {log_noise_scale:.2f}")
        elif key == ord('d'):
            log_noise_scale = max(0, log_noise_scale - 0.01)
            print(f"Noise scale decreased to {log_noise_scale:.2f}")
        elif key == ord('l'):
            low_pass_on = not low_pass_on
            print(f"Low-pass filter toggled to {'on' if low_pass_on else 'off'}")
        elif key == ord('s'):
            strength += 0.5
            print(f"Strength increased to {strength:.2f}")
        elif key == ord('r'):
            strength = max(0.5, strength - 0.5)
            print(f"Strength decreased to {strength:.2f}")
        elif key == ord('z'):
            log_noise_scale = 0.77
            strength = 1.5
            filter_on = True
            pyramid_on = True
            low_pass_on = False
            noise_method = compute_noise_s
            print("Reset settings")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
