from ultralytics import YOLO
import torch
import numpy as np

# Cek apakah GPU tersedia
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model hasil training
model_path = 'yolov8n.pt'  # Pastikan model ini sudah ada di direktori Colab
model = YOLO(model_path).to(device)  # Memindahkan model ke GPU jika tersedia

# Evaluasi model pada dataset validasi
metrics = model.val(data="dataset/data.yaml", split="val")


# Cetak hasil evaluasi
print(f"Precision: {np.mean(metrics.box.p):.4f}")  # Calculate and print mean precision
print(f"Recall: {np.mean(metrics.box.r):.4f}")  # Calculate and print mean recall
print(f"F1-score: {np.mean(metrics.box.f1):.4f}")  # Calculate and print mean F1-score
print(f"mAP50: {metrics.box.map50:.4f}")  # mAP pada IoU 0.5
print(f"mAP50-95: {metrics.box.map:.4f}")  # mAP rata-rata IoU 0.5-0.

