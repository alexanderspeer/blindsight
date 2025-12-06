import cv2

cap = cv2.VideoCapture("http://10.207.55.64:8090/stream.mjpg")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("pi camera", frame)
    if cv2.waitKey(1) == 27:
        break
