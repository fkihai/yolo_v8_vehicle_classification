import cv2

# RTSP URL for your camera
RTSP_URL = "rtsp://admin:admin@192.168.1.100:8554/Streaming/Channels/101"

# Initialize video capture from the RTSP stream
cap = cv2.VideoCapture(RTSP_URL)

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

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
