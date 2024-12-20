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

# Model Yolo
# doc : https://docs.ultralytics.com/models
model_path = os.path.join('.', 'yolov5nu.pt')

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
filter_object = ['car', 'bus', 'bicycle','truck','motorbike']
filter_ids = [2, 5, 1, 7, 3]

# millis update object calculate
interval = 100
previous_time = int(round(time.time() * 1000))

# conting
object_counting = 0
classification = ""

# ever detect
vehicle_detect = []

def vehicle_car_classification(centimer):
    if centimer <= 500:
        return "IV B"
    elif 500 < centimer <= 770:
        return "V B"
    elif 770 < centimer <= 1000:
        return "VI B"
    elif 1000 < centimer <= 1200:
        return "VII B"
    elif 1200 < centimer <= 1600:
        return "VIII B"
    else:
        return "IX B"

def calculate_bounding_box_width(x1, x2):
    width = int(x2 - x1)
    return width

def pixel_to_centimeters(pixel_width):
    # 358 pixel  = 7.4 meter -> calibration
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

# RTSP stream sebagai input
# stream_url = "rtsp://username:password@192.168.1.100:554/stream"

while ret:

    # hitung fps
    frame_counter += 1
    if frame_counter >= 10:  # Hitung setiap 10 frame untuk stabilitas
        elapsed_time = time.time() - start_time
        fps = frame_counter / elapsed_time
        frame_counter = 0
        start_time = time.time()

    results = results = model.track(
        source=frame,       
        persist=True,             
        conf=0.5,                
        device="0",               
        save=False,               
        classes=filter_ids,            
        show=False                
    )  
    
    # Buat Area Garis Merah
    cv2.rectangle(frame, (int(x1_frame), int(y1_frame)), (int(x2_frame), int(y2_frame)), (0, 255, 0), 4)

    # Buat garis hitung
    cv2.line(frame, (line_crossing_x_start, line_crossing_y), (line_crossing_x_end, line_crossing_y), (255, 0, 0), 4)


    # Draw bounding boxes and labels for detections
    for result in results:
        try:
            for box in result.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    id = int(box.id)  # ID unik dari pelacakan
                    cls = result.names[int(box.cls)]  # Nama kelas objek
                    conf = float(box.conf)  # Confidence score

                    # Deteksi on ROI (Region of Interest)
                    if x1_frame <= x1 <= x2_frame and y1_frame <= y1 <= y2_frame:
                        try:
                            if is_crossing_line(y2) and id not in vehicle_detect:
                                object_counting += 1
                                vehicle_detect.append(id)
                                pixel_width = calculate_bounding_box_width(x1, x2)
                                centimeter = pixel_to_centimeters(pixel_width)
                                classification = vehicle_car_classification(centimeter)
                                inserData(cls, pixel_width, centimeter, classification)
                        except Exception as e:
                            print(f"Error during vehicle detection: {e}")

                        # Visualisasi bounding box dan label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{cls} ID:{id} Conf:{conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error processing bounding box: {e}")
        except Exception as e:
            print(f"Error processing result: {e}")


    # print counter
    cv2.putText(frame, f"Count : {object_counting}", (900, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Class : {classification}", (900, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Menampilkan FPS pada frame
    cv2.putText(frame, f"FPS : {fps:.2f}", (900, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)


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

