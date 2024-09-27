GNAPing On the Job: Attacking and Defending Facial Detection on Edge Devices

Welcome to the repository for GNAP Attack and Defense—a collection of Python scripts used to implement and test adversarial attacks and defenses on facial detection systems, specifically targeting edge devices. This repository accompanies the research paper "GNAPing On the Job: Attacking and Defending Facial Detection on Edge Devices" (currently under review).

Table of Contents
Overview
Folder Structure
Getting Started
Usage
Attack Scripts
Defense Scripts
Contributing
License
Overview
This repository provides the code for implementing the Guided-inspired Noise Attack Pyramid (GNAP) and its defense counterpart, the Guided Noise Attack Guard (GNAG). Both techniques are designed for facial recognition systems deployed on resource-constrained devices (edge devices). The goal is to simulate real-world adversarial scenarios and explore defense mechanisms that maintain system robustness and performance.

Attack Overview
GNAP: A signal-dependent noise attack that uses the Laplacian-of-Gaussian (LoG) and Laplacian Pyramid filters to reduce the accuracy of facial detection systems by perturbing critical features, such as facial landmarks.
Defense Overview
GNAG: A defense mechanism utilizing the Unsharp Mask filter, designed to restore image quality and maintain detection accuracy in the presence of adversarial attacks.
Folder Structure
The repository is structured as follows:

bash
Copy code
GNAP_Attack_Defense/
│
├── attack/                         
│   ├── attack_images.py            # General attack on still images
│   ├── attack_lfw.py               # Attack + defense on the LFW dataset with confidence calculations
│   ├── image_attacker.py           # Attack using YOLO model for confidence calculation
│   └── laplace_fps.py              # Real-time attack with FPS calculation
│
├── defense/                        
│   ├── defend_image.py             # Apply defense to already attacked images
│   ├── clean_image.py              # Adjust defense parameters dynamically on clean images
Getting Started
Prerequisites
To get started, you’ll need the following dependencies installed:

Python 3.6+
TensorFlow or PyTorch (depending on your model)
OpenCV
NumPy
You can install the dependencies using the following command:

bash
Copy code
pip install -r requirements.txt
Dataset
To reproduce the experiments from the paper, you can use the Labeled Faces in the Wild (LFW) dataset, available here.

Usage
Attack Scripts
attack_images.py: This script applies the GNAP attack on a set of still images. You can specify the input directory containing images to attack.

Example usage:

bash
Copy code
python attack_images.py --input_dir /path/to/images --output_dir /path/to/save
attack_lfw.py: Performs both attack and defense on the LFW dataset. Outputs include the modified images and calculated confidence scores.

Example usage:

bash
Copy code
python attack_lfw.py --lfw_dir /path/to/lfw --output_dir /path/to/save
image_attacker.py: Uses a YOLO model to calculate confidence scores after applying the attack.

laplace_fps.py: This script runs a real-time attack on video input, measuring FPS and attack impact.

Defense Scripts
defend_image.py: Apply the GNAG defense on images that have already been attacked. This can be used to restore the image quality and accuracy of the model.

Example usage:

bash
Copy code
python defend_image.py --input_dir /path/to/attacked_images --output_dir /path/to/save
clean_image.py: Adjusts the defense dynamically for clean images.

Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue if you find bugs or want to suggest improvements.

License
This project is licensed under the MIT License - see the LICENSE file for details.
