import os
from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'truck1-video.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

# model_path = os.path.join('.', 'runs', 'detect', 'train8', 'weights', 'best.pt')
model_path = os.path.join('.', 'model/yolov8n.pt')

# Load the model
model = YOLO(model_path)  # Load a custom YOLO model

threshold = 0.50

width = 1280   # Lebar frame 
height = 720   # Tinggi frame 

x1_frame, y1_frame = 80, 400  # Pojok kiri atas
x2_frame, y2_frame = 1600, 1000  # Pojok kanan bawah

def calculateBox(x1, y1, x2, y2):
    width = int(x2 - x1)
    height = int(y2 - y1)
    area = width * height
    return area

while ret:
    # Run inference on the frame
    results = model(frame)[0]
    
    # Draw bounding boxes and labels for detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper() + " " + str(round(score, 2)), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # resize image
    frame = cv2.resize(frame, (width, height))
    # Display the frame with detections
    cv2.imshow('YOLO Object Detection', frame)

    # Check if the user presses the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

