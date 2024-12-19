import os
from ultralytics import YOLO
import cv2
import time

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'car-video.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

model_path = os.path.join('.', 'yolov8m.pt')

# Load the model
model = YOLO(model_path)  # Load a custom YOLO model

threshold = 0.50

width = 1280   # Lebar frame 
height = 720   # Tinggi frame 

# Gambar Kotak Merah
x1_frame, y1_frame = 50, 250  # Pojok kiri atas
x2_frame, y2_frame = 1400, 850  # Pojok kanan bawah

# Garis Start dan End
line_crossing = 800
line_x_start, line_x_end = 20,1900

# Pelacakan objek
object_count_car = 0
object_count_bus = 0
object_count_truck = 0

# penampungan id detect
car_tracked_ids = set()
bus_tracked_ids = set()
truck_tracked_ids = set()

filter_object = ['car', 'bus', 'bicycle','truck']

# id Kendaraan
car_id = 0
bus_id = 0
truck_id = 0

# Fungsi untuk memeriksa apakah bounding box melintasi garis
def is_crossing_line(box):
    x1,y1,x2,y2 = box
    box_center_y = (y1 + y2) / 2
    box_center_x = (x1 + x2) / 2

    if line_crossing - 3 <= box_center_y <= line_crossing + 3 and line_x_start <= box_center_x <= line_x_end:
        return True
    return False 


while ret:
    # Run inference on the frame
    results = model(frame)[0]
    
    # Buat Area Garis Merah
    # cv2.rectangle(frame, (int(x1_frame), int(y1_frame)), (int(x2_frame), int(y2_frame)), (0, 0, 255), 4)

    # Gambar garis start dan end
    cv2.line(frame, (line_x_start, line_crossing), (line_x_end, line_crossing), (0, 255, 0), 2)

    # Draw bounding boxes and labels for detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        # Deteksi on ROI (Region of Interest)
        if score > threshold and str(results.names[class_id]) in filter_object:
            
            if is_crossing_line([x1,y1,x2,y2]):
                if str(results.names[class_id]) == "car":
                    car_id = f"{int(x1)}-{int(y1)}-{int(x2)}-{int(y2)}"
                    if car_id not in car_tracked_ids:
                        car_tracked_ids.add(car_id)
                        object_count_car += 1
                if str(results.names[class_id]) == "bus":
                    bus_id = f"{int(x1)}-{int(y1)}-{int(x2)}-{int(y2)}"
                    if bus_id not in bus_tracked_ids:
                        bus_tracked_ids.add(bus_id)
                        object_count_bus += 1
                if str(results.names[class_id]) == "truck":
                    truck_id = f"{int(x1)}-{int(y1)}-{int(x2)}-{int(y2)}"
                    if truck_id not in truck_tracked_ids:
                        truck_tracked_ids.add(truck_id)
                        object_count_truck += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, str(results.names[class_id]).upper() + " " + str(round(score, 2)), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
            cv2.putText(frame, "cars : " + str(object_count_car), (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "bus : " + str(object_count_bus), (50,130), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "truck : " + str(object_count_truck), (50,180), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
                    

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

def calculateBox( x1, y1, x2, y2):
    width = int(x2 - x1)
    height = int(y2 - y1)
    area = width * height
    return area

