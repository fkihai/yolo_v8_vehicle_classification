import cv2

def open_camera(index):
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        print(f"Kamera {index} tidak tersedia.")
        return None
    return cam

# Coba buka kamera sekali di luar loop
cam = open_camera(2)

if cam is None:
    print("Gagal membuka kamera. Program berhenti.")
else:
    try:
        while True:
            ret, frame = cam.read()

            if not ret:
                print("Gagal membaca frame dari kamera.")
                break

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error capturing images: {e}")

    finally:
        cam.release()  # Pastikan kamera dilepas saat keluar dari loop
        cv2.destroyAllWindows()
