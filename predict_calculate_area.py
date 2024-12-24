import os
from ultralytics import YOLO,solutions
import cv2
import time
import mysql.connector

VIDEOS_DIR = os.path.join('.', 'videos')
video_path = os.path.join(VIDEOS_DIR, 'truck-test.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

# variable untuk menghitung vps
frame_counter = 0
fps = 0
start_time = time.time()

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
interval = 100
previous_time = int(round(time.time() * 1000))

# conting
object_conting = 0

# ever detect
vehicle_centroids = []

def vehicle_car_classification(centimer):
    if centimer <= 600:
        return "IV B"
    elif 600 < centimer <= 800:
        return "V B"
    elif 800 < centimer <= 1100:
        return "VI B"
    elif 1100 < centimer <= 1300:
        return "VII B"
    elif 1300 < centimer <= 1700:
        return "VIII B"
    else:
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
    if line_crossing_y - 5 <= y2 <= line_crossing_y + 5:
        return True
    return False

def calculate_centroid(x1,y1,x2,y2):
    return ((x1 + x2) // 2,(y1 + y1) // 2)

def is_similiar_object(new_centroid,existing_centroid,max_distance=20):
    for centroid in existing_centroid:
        distance = ((new_centroid[0] - centroid[0])**2 + (new_centroid[1]-centroid[1])**2)
        if distance < max_distance :
            return True
    return False 

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
                    if is_crossing_line(y2) :
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
    # Menampilkan FPS pada frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (900, 100),
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

