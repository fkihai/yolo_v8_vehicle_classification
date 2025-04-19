from ultralytics import YOLO
from imutils.video import VideoStream
import mysql.connector
import cv2
import time
import imutils
import os

RTSP_URL_SD = "rtsp://admin:admin@192.168.0.100:8554/Streaming/Channels/102"

VIDEOS_DIR = os.path.join('.', 'videos')
VIDEO_PATH = os.path.join(VIDEOS_DIR, 'truck-test.mp4')

cap = cv2.VideoCapture(RTSP_URL_SD)
time.sleep(2)
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
model_path = os.path.join('.', 'yolov8n.pt')

# Load the model
model = YOLO(model_path)  # Load a custom YOLO model
threshold = 0.60

# variable calibration 
width_in_pixel = 358
width_in_meter = 7.4

height_in_pixel = 358
height_in_meter = 2.1


# convert ke settingan 640 x 360
cap_height = int(frame.shape[0])
cap_width = int(frame.shape[1])

mutliple_width = int(cap_width / 640)
mutliple_height = int(cap_height / 360)

# Resize Frame show
width = 1080  
height = 640  

# ROI
x1_frame, y1_frame = 35 * mutliple_width, 70 * mutliple_height  # Pojok kiri atas
x2_frame, y2_frame = 450 * mutliple_width, 290 * mutliple_height  # Pojok kanan bawah

# Batas Garis Hitung 
start_line_crossing_y = 235 * mutliple_height
end_line_crossing_y = 255 * mutliple_height

start_line_crossing_x1, start_line_crossing_x2 = 40 * mutliple_width, 445 * mutliple_width
end_line_crossing_x1, end_line_crossing_x2 = 40 * mutliple_width, 445 * mutliple_width

# bool hitung
cross_start = False

# object detection
filter_object = ['car', 'bus', 'bicycle','truck','motorbike','person']
filter_ids = [2, 5, 1, 7, 3]

# conting
object_counting = 0
classification = ""

height_cm = 0
width_cm = 0

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

def calculate_bounding_box_width(x1, x2, y1, y2):
    width = int(x2 - x1)
    height = int(y2 - y1)
    return height, width

def width_pixel_to_cm(pixel_width):    
    centimeter = pixel_width * ((width_in_meter * 100) / width_in_pixel)
    return centimeter;

def height_pixel_to_cm(pixel_height):    
    centimeter = pixel_height * ((height_in_meter * 100) / height_in_pixel)
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

while ret:

    # hitung fps
    frame_counter += 1
    if frame_counter >= 10:  # Hitung setiap 10 frame untuk stabilitas
        elapsed_time = time.time() - start_time
        fps = frame_counter / elapsed_time
        frame_counter = 0
        start_time = time.time()

    results = model.track(
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
    cv2.rectangle(frame, (int(x1_frame), int(y1_frame)), (int(x2_frame), int(y2_frame)), (0, 255, 0), 2)

    # Buat garis hitung
    cv2.line(frame, (start_line_crossing_x1, start_line_crossing_y), (start_line_crossing_x2, start_line_crossing_y), (0, 0, 255), 2)
    cv2.line(frame, (end_line_crossing_x1, end_line_crossing_y), (end_line_crossing_x2, end_line_crossing_y), (255, 0, 0), 2)


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

                                pixel_height, pixel_width = calculate_bounding_box_width(x1, x2,y1, y2)

                                height_cm = height_pixel_to_cm(pixel_height)
                                width_cm = width_pixel_to_cm(pixel_width)

                                classification = vehicle_car_classification(cls,width_cm)
                                inserData(cls, pixel_height, pixel_width, height_cm, width_cm, classification)

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


    cv2.putText(frame, f"Count : {object_counting}", (500 * mutliple_width, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Class : {classification}", (500 * mutliple_width, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Height (m): {height_cm/100:.1f}", (500 * mutliple_width, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Length (m) : {width_cm/100:.1f}", (500 * mutliple_width, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

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
