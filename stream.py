import cv2
import imutils
from imutils.video import VideoStream

ATCS = "https://atcsdishub.pemkomedan.go.id/camera/PANDUCIREBON.m3u8"

video_stream = VideoStream(ATCS).start()

while True:
    frame = video_stream.read()
    if frame is None:
        print("Error: Failed to read frame from stream.")
        break
    
    # Display the frame
    frame = imutils.resize(frame,width=720)
    cv2.imshow("Camera Feed", frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_stream.stop()
cv2.destroyAllWindows()
