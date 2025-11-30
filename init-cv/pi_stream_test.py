import cv2

url = "http://127.0.0.1:8890/"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: cannot open stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("no frame")
        continue

    cv2.imshow("Pi Camera (MJPEG over SSH Tunnel)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
