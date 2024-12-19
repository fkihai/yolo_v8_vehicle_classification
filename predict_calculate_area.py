import os
from ultralytics import YOLO
import cv2
import time
import mysql.connector

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'truck-test.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

#  DATABASE
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database = "kelas_kendaraan"
)
mycursor = mydb.cursor()

# model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
model_path = os.path.join('.', 'yolov8m.pt')

# Load the model
model = YOLO(model_path)  # Load a custom YOLO model
threshold = 0.60

# Resize Frame
width = 1280   # Lebar frame 
height = 720   # Tinggi frame 

# ROI
x1_frame, y1_frame = 150, 180  # Pojok kiri atas
x2_frame, y2_frame = 1100, 650  # Pojok kanan bawah

# Batas Garis Hitung 
line_crossing_y = 640
line_crossing_x_start, line_crossing_x_end = 160,1090

# object detection
filter_object = ['car', 'bus', 'bicycle','truck']

# millis update object calculate
interval = 50
previous_time = int(round(time.time() * 1000))

# conting
object_conting = 0

### info ####
# 358 pixel  = 7.4 meter
# 1 pixel = 2.067  cm

def vehicle_car_classification(centimer):
    # class IV B
    if centimer < 500 :
        return "IV B"
    # class V B
    if  500 > centimer <= 700 :
        return "V B"
    # class VI B
    if  700 > centimer <= 1000 :
        return "VI B"
    # class VII B
    if  1000 > centimer <= 1200 :
        return "VI B"
    # class VIII B
    if  1200 > centimer <= 1600 :
        return "VI B"
    # class IX B
    if  centimer > 1600 :
        return "IX B"

def calculate_bounding_box_width(x1, x2):
    width = int(x2 - x1)
    return width

def pixel_to_centimeters(pixel_width):
    centimeter = pixel_width * 2.067
    return centimeter;

def inserData(name, pixel, centimeter, classification ):
    sql = """
    INSERT INTO result (name, long_pixel, long_cm, classification) 
    VALUES (%s , %s, %s, %s)
    """
    val = (name, pixel, centimeter,classification)
    mycursor.execute(sql,val)
    mydb.commit()

# Fungsi untuk memeriksa apakah bounding box melintasi garis
def is_crossing_line(y2):
    if line_crossing_y - 3 <= y2 <= line_crossing_y + 3 :
        return True
    return False

while ret:
    # Run inference on the frame
    results = model(frame)[0]
    
    # Buat Area Garis Merah
    cv2.rectangle(frame, (int(x1_frame), int(y1_frame)), (int(x2_frame), int(y2_frame)), (0, 255, 0), 4)

    # Buat garis hitung
    cv2.line(frame, (line_crossing_x_start, line_crossing_y), (line_crossing_x_end, line_crossing_y), (255, 0, 0), 4)

    # Draw bounding boxes and labels for detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        # Deteksi on ROI (Region of Interest)
        if x1_frame <= x1 <= x2_frame and y1_frame <= y1 <= y2_frame:
            # filter object detection
            if score > threshold and results.names[int(class_id)] in filter_object:

                current_time = int(round(time.time() * 1000))
                if current_time - previous_time >= interval:
                    previous_time = current_time

                    # condition
                    if is_crossing_line(y2):
                        object_conting += 1
                        pixel_width = calculate_bounding_box_width(x1,x2)
                        centimeter = pixel_to_centimeters(pixel_width)
                        classification = vehicle_car_classification(centimeter)
                        inserData(results.names[int(class_id)],pixel_width,centimeter,classification)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper() + " " + str(calculate_bounding_box_width(x1,x2)), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    
    
    # print counter
    cv2.putText(frame, f"counting : {object_conting}", (900, 50),
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

