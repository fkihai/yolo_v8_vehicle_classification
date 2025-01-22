import os
from ultralytics import YOLO
import cv2
import time

RTSP_URL = "rtsp://admin:admin@192.168.1.100:8554/Streaming/Channels/101"

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'truck-test.mp4')

cap = cv2.VideoCapture(RTSP_URL)
ret, frame = cap.read()
H, W, _ = frame.shape

# variable untuk menghitung vps
frame_counter = 0
fps = 0
start_time = time.time()

model_path = os.path.join('.', 'yolov8m.pt')

# Load the model
model = YOLO(model_path) 
threshold = 0.50

width = 1280   # Lebar frame 
height = 720   # Tinggi frame 

# limit detection
filter_object = ["truck",'car','person']

while ret:

    # hitung fps
    frame_counter += 1
    if frame_counter >= 10:  # Hitung setiap 10 frame untuk stabilitas
        elapsed_time = time.time() - start_time
        fps = frame_counter / elapsed_time
        frame_counter = 0
        start_time = time.time()

    # Run inference on the frame
    results = model(frame)[0]
    
    # Draw bounding boxes and labels for detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if score > threshold and results.names[int(class_id)] in filter_object:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper() + " " + str(round(score, 2)), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Menampilkan FPS pada frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (1000, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # resize image
    frame = cv2.resize(frame, (width, height))

    # Display the frame with detections
    cv2.imshow('Test Object Detection', frame)

    # Check if the user presses the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

