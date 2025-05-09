# 🚢 Ferry Vehicle Detection and Classification with YOLOv8

This project utilizes **YOLOv8m** from the **Ultralytics** library to detect and classify vehicles for ticketing purposes on ferry transportation. It identifies vehicle types such as cars, trucks, buses, and motorcycles from the **COCO dataset**.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [How to Run](#how-to-run)
6. [Code Structure](#code-structure)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Acknowledgements](#acknowledgements)

---

## 🔍 Overview

The project uses **YOLOv8m** for object detection and classification of vehicles entering a ferry boarding station. The goal is to automate ticket management based on vehicle types detected in a real-time video stream or image input.

**Key Vehicle Classes** (from COCO dataset):

- 🚗 Car
- 🚚 Truck
- 🛵 Motorcycle
- 🚌 Bus

The detected vehicle dimensions (in pixels) are converted to real-world measurements to assist in ticket classification.

---

## ✨ Features

- **Vehicle Detection**: Real-time detection of vehicles using YOLOv8m.
- **Classification**: Classifies vehicles into specific categories (e.g., car, truck, bus, motorcycle).
- **Bounding Box Metrics**: Calculates pixel width of the bounding box and converts it to meters.
- **Easy Integration**: Built with Python and Ultralytics for ease of deployment.
- **Customizable**: Scalable for other YOLO versions and datasets.

---

## 🛠️ Requirements

Ensure the following dependencies are installed:

- Python 3.10+
- pip 22.0+
- Ultralytics
- mysql-connector-python
- imutils

---

## 🚀 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/fkihai/yolo_v8_vehicle_classification.git
   cd yolo_v8_vehicle_classification
   ```

2. **Set up the environment**:
   Install required libraries:

   ```bash
   pip install ultralytics opencv-python numpy matplotlib
   ```

3. **Download YOLOv8m model**:
   - The pre-trained YOLOv8m model can be downloaded from [Ultralytics YOLOv8 Models](https://github.com/ultralytics/ultralytics).
   - Save it in the project directory as `yolov8m.pt`.

---

## ▶️ How to Run

1. **Run the detection script**:
   Use the following command to process an image or video:

   ```bash
   python detect.py --source path/to/your/video_or_image --weights yolov8m.pt
   ```

   Example:

   ```bash
   python detect.py --source test_video.mp4 --weights yolov8m.pt
   ```

2. **Optional Arguments**:

   - `--source`: Path to the input video/image.
   - `--weights`: Path to the YOLOv8 model file.
   - `--conf`: Confidence threshold for detection (default: 0.25).

3. **Output**:
   - The processed video or image will be saved in the `runs/detect` folder with bounding boxes and vehicle classifications.

---

## 📂 Code Structure

```plaintext
ferry-vehicle-detection/
│
├── detect.py          # Main script for vehicle detection
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
├── yolov8m.pt         # YOLOv8 model weights (downloaded)
└── data/
    ├── test_video.mp4 # Example input video
    └── output/        # Directory for saving outputs
```

---

## 📊 Results

Example detection output:

![Example Detection](https://via.placeholder.com/600x300?text=Detection+Output+Example)

| Vehicle Type | Pixel Width | Converted Length (meters) |
| ------------ | ----------- | ------------------------- |
| Car          | 358 px      | 7.4 m                     |
| Truck        | 512 px      | 10.6 m                    |
| Motorcycle   | 120 px      | 2.5 m                     |

---

## 🔮 Future Work

- Add a real-world calibration feature for more accurate size estimation.
- Integrate a database to store detection results for ticket management.
- Deploy the system on edge devices like Raspberry Pi for real-time use.

---

## 🙏 Acknowledgements

- [Ultralytics](https://ultralytics.com/) for the YOLOv8 implementation.
- COCO Dataset for pre-trained object classes.

---

## 📜 License

This project is licensed under the MIT License.

---

**Author**: [Your Name](https://github.com/yourusername)  
**Contact**: your_email@example.com

---

Feel free to replace placeholders (e.g., links, your name) with relevant information! 😊
