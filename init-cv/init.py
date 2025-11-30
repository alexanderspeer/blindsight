#you're going to want to run the cmd:
#rpicam-vid -t 0 --codec mjpeg -o - | ffmpeg -i - -f mpjpeg -listen 1 http://0.0.0.0:8090/

#then you can run this script to view the camera feed:

import cv2

cap = cv2.VideoCapture("udp://@:5000")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("stream", frame)
    if cv2.waitKey(1) == 27:
        break
