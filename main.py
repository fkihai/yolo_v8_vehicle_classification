import os
import subprocess

# 1. Latih Model YOLOv8
def train_model():
    try:
        # Menggunakan YOLOv8 untuk melatih model dengan konfigurasi yang sesuai
        subprocess.check_call([
            "yolo", "train",
            "model=yolov8n.pt",  # Menggunakan model YOLOv8 Nano sebagai model dasar
            "data=config.yaml",  # File konfigurasi data.yaml
            "epochs=100",  # Jumlah epoch (Anda bisa sesuaikan)
            "batch=16",  # Ukuran batch
            "imgsz=640"  # Ukuran gambar input
        ])
        print("Pelatihan model selesai.")
    except subprocess.CalledProcessError:
        print("Pelatihan model gagal. Periksa error di konsol.")
        return False
    return True

# 4. Evaluasi Model
def evaluate_model():
    try:
        # Evaluasi model yang sudah dilatih
        subprocess.check_call([
            "yolo", "val",
            "model=runs/detect/train/weights/best.pt",  # Jalur model terbaik yang telah dilatih
            "data=config.yaml"  # File konfigurasi data
        ])
        print("Evaluasi model selesai.")
    except subprocess.CalledProcessError:
        print("Evaluasi model gagal.")
        return False
    return True

# 5. Prediksi dengan Model
def make_predictions():
    try:
        # Prediksi dengan model terlatih
        subprocess.check_call([
            "yolo", "detect", "predict",
            "model=runs/detect/train/weights/best.pt",  # Jalur model terbaik
            "source=C:/Users/msila/Documents/project/yolo_v8_custom_model/data/test/images"  # Jalur folder gambar test
        ])
        print("Prediksi selesai.")
    except subprocess.CalledProcessError:
        print("Prediksi gagal.")
        return False
    return True


# Main Function
def main():
    # print("Langkah 1: Melatih model YOLOv8...")
    # if not train_model():
    #     return

    # print("Langkah 2: Evaluasi model...")
    # if not evaluate_model():
    #     return

    print("Langkah 3: Prediksi dengan model terlatih...")
    make_predictions()

if __name__ == "__main__":
    main()
