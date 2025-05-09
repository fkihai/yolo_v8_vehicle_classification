# FIKSS

import os
from ultralytics import YOLO
import cv2
import time
import mysql.connector

RTSP_URL = "rtsp://admin:admin@192.168.0.100:8554/Streaming/Channels/101"

VIDEOS_DIR = os.path.join('videos')
VIDEO_PATH = os.path.join(VIDEOS_DIR, 'truck-test.mp4')

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Gagal membuka video di path {VIDEO_PATH}")
    exit()

ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Gagal membaca frame dari video.")
    exit()

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
model_path = os.path.join('.', 'yolov8m.pt')

# Load the model
model = YOLO(model_path)  # Load a custom YOLO model
threshold = 0.60


# convert ke settingan 1280 x 720
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mutliple_width = int(cap_width / 1280)
mutliple_height = int(cap_height / 720)


# Resize Frame show
width = 1280  
height = 720  


# ROI
x1_frame, y1_frame = 190 * mutliple_width, 180 * mutliple_height  # Pojok kiri atas
x2_frame, y2_frame = 1300 * mutliple_width, 650 * mutliple_height  # Pojok kanan bawah

# Batas Garis Hitung 
start_line_crossing_y = 540 * mutliple_height
end_line_crossing_y = 640 * mutliple_height

start_line_crossing_x1, start_line_crossing_x2 = 200 * mutliple_width,1290 * mutliple_width
end_line_crossing_x1, end_line_crossing_x2 = 200 * mutliple_width, 1290 * mutliple_width

# bool hitung
cross_start = False

# object detection
filter_object = ['car', 'bus', 'bicycle','truck','motorbike','person']
filter_ids = [2, 5, 1, 7, 3]

# conting
object_counting = 0
classification = ""

height_cm = 0
length_cm = 0

def vehicle_car_classification(classification,centimer):
    if classification != "motorcycle" and classification != "bicycle":
        if centimer <= 500:
            return "IV B"
        elif 500 < centimer <= 700:
            return "V B"
        elif 700 < centimer <= 1000:
            return "VI B"
        elif 1000 < centimer <= 1200:
            return "VII B"
        elif 1200 < centimer <= 1600:
            return "VIII B"
        else:
            return "IX B"
    elif classification == "motorcycle" : return "II B"
    elif classification == "bicycle" : return "I B"


def calculate_bounding_box_length(value1, value2):
    result = int(value2 - value1)
    return result

def pixel_length_to_centimeters(pixel_width):
    # 353 pixel  = 5.9 meter -> calibration -> 590 cm / 353 -> 1.671
    centimeter = pixel_width * 1.671
    return centimeter;

def pixel_height_to_centimeters(pixel_height):
    # 390 pixel  = 2.5 meter -> calibration -> 250 cm / 390 - > 0.641
    centimeter = pixel_width * 0.641
    return centimeter;

def inserData(name, height_pixel, widht_pixel, height_cm, width_cm, classification ):
    sql = """
    INSERT INTO result (name, height_pixel, width_pixel, height_cm, width_cm, classification) 
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    val = (name, widht_pixel, height_pixel,height_cm, width_cm, classification)
    mycursor.execute(sql,val)
    mydb.commit()

# Fungsi untuk memeriksa apakah bounding box melintasi garis
def is_start_crossing_line(y2):
    if start_line_crossing_y - 5 <= y2 <= start_line_crossing_y + 5:
        return True
    return False

def is_end_crossing_line(y2):
    if end_line_crossing_y - 5 <= y2 <= end_line_crossing_y + 5:
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
        persist=False,             
        conf=0.5,                
        device="0",               
        save=False,               
        classes=filter_ids,            
        show=False,
        stream=True               
    )  
    
    # Buat Area deteksi
    cv2.rectangle(frame, (int(x1_frame), int(y1_frame)), (int(x2_frame), int(y2_frame)), (0, 255, 0), 4)

    # Buat garis hitung
    cv2.line(frame, (start_line_crossing_x1, start_line_crossing_y), (start_line_crossing_x2, start_line_crossing_y), (0, 0, 255), 4)
    cv2.line(frame, (end_line_crossing_x1, end_line_crossing_y), (end_line_crossing_x2, end_line_crossing_y), (255, 0, 0), 4)


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
                            if is_start_crossing_line(y2) : cross_start = True
                            if cross_start and is_end_crossing_line(y2) :
                                cross_start = False
                                object_counting += 1

                                pixel_height = calculate_bounding_box_length(y1, y2)
                                pixel_width = calculate_bounding_box_length(x1, x2)
                                
                                height_cm = pixel_height_to_centimeters(pixel_height)
                                length_cm = pixel_length_to_centimeters(pixel_width)

                                classification = vehicle_car_classification(cls,length_cm)
                                inserData(cls, pixel_height, pixel_width, height_cm, length_cm, classification)

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


    cv2.putText(frame, f"Count : {object_counting}", (900 * mutliple_width, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Class : {classification}", (900 * mutliple_width, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Height (m) : {height_cm/100:.2f}", (900 * mutliple_width, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Length (m) : {length_cm/100:.2f}", (900 * mutliple_width, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    # resize image
    frame = cv2.resize(frame, (width, height)
                       )
    # Display the frame with detections
    cv2.imshow('YOLO Object Detection', frame)

    # Check if the user presses the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()