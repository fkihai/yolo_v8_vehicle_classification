import os
from ultralytics import YOLO
import cv2
import time
import mysql.connector

RTSP_URL = "rtsp://admin:admin@192.168.1.100:8554/Streaming/Channels/101"

VIDEOS_DIR = os.path.join('.', 'videos')
VIDEO_PATH = os.path.join(VIDEOS_DIR, 'truck-test.mp4')

cap = cv2.VideoCapture(VIDEO_PATH)
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
x1_frame, y1_frame = 150 * mutliple_width, 180 * mutliple_height  # Pojok kiri atas
x2_frame, y2_frame = 1100 * mutliple_width, 650 * mutliple_height  # Pojok kanan bawah

# Batas Garis Hitung 
start_line_crossing_y = 540 * mutliple_height
end_line_crossing_y = 640 * mutliple_height

start_line_crossing_x1, start_line_crossing_x2 = 160 * mutliple_width,1090 * mutliple_width
end_line_crossing_x1, end_line_crossing_x2 = 160 * mutliple_width, 1090 * mutliple_width

# bool hitung
cross_start = False

# object detection
filter_object = ['car', 'bus', 'bicycle','truck','motorbike','person']
filter_ids = [2, 5, 1, 7, 3]

# conting
object_counting = 0
classification = ""
centimeter = 0

def vehicle_car_classification(classification,centimer):
    if classification != "motorcycle" and classification != "bicycle":
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
    elif classification == "motorcycle" : return "II B"
    elif classification == "bicycle" : return "I B"

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
                                pixel_width = calculate_bounding_box_width(x1, x2)
                                centimeter = pixel_to_centimeters(pixel_width)
                                classification = vehicle_car_classification(cls,centimeter)
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


    cv2.putText(frame, f"Count : {object_counting}", (900 * mutliple_width, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Class : {classification}", (900 * mutliple_width, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f"meter : {centimeter/100:.2f}", (900 * mutliple_width, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

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

