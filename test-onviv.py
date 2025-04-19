from ultralytics import YOLO
import os
import cv2

# RTSP URL for your camera
RTSP_URL = "rtsp://admin:admin@192.168.1.100:8554/Streaming/Channels/101"
ATCS = "https://atcsdishub.pemkomedan.go.id/camera/PANDUCIREBON.m3u8"

# Initialize video capture from the RTSP stream
cap = cv2.VideoCapture(ATCS)

# Load the model
model_path = os.path.join('.', 'yolov8n.pt')
model = YOLO(model_path) 
threshold = 0.70


# Check if the camera is accessible
if not cap.isOpened():
    print("Error: Cannot open RTSP stream.")
    exit()

# Read and display frames in a loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from stream.")
        break
    
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper() + " " + str(round(score, 2)), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)


    # Display the frame
    resize = cv2.resize(frame,(1270,720))
    cv2.imshow("Camera Feed", resize)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
