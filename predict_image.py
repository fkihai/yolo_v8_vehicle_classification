import os
from ultralytics import YOLO
import cv2

# Path ke gambar
IMAGE_DIR = os.path.join('.','images')
image_path = os.path.join(IMAGE_DIR, 'truck-tambang.jpg')

# Validasi file gambar
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at {image_path}")

# Path ke model YOLO
model_path = os.path.join('.','yolov8m.pt')

# Load model YOLO
model = YOLO(model_path)

# Threshold deteksi
threshold = 0.50

# resize image
width,height = 1280,720 

# Baca gambar
image = cv2.imread(image_path)
if image is None:
    raise RuntimeError("Failed to read the image file. Please check the file path or format.")

# Jalankan inferensi pada gambar
results = model(image)[0]

# Gambar kotak deteksi dan label
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        label = f"{results.names[int(class_id)].upper()} {round(score,2)}"
        cv2.putText(image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)


# resize image
image = cv2.resize(image, (width, height))

# Tampilkan gambar dengan deteksi
cv2.imshow('Test Object Detection', image)
cv2.waitKey(0)  # Tunggu hingga tombol ditekan
cv2.destroyAllWindows()
