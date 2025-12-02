#this is how you run the ssh pi:

# rpicam-vid \
#     --codec h264 \
#     --profile high \
#     --level 4.2 \
#     --width 1280 \
#     --height 720 \
#     --framerate 30 \
#     --bitrate 8000000 \
#     --inline \
#     --intra 30 \
#     --timeout 0 \
#     --listen \
#     -o tcp://0.0.0.0:5001


# then once you have run that command on ssh speeriocheerio@<PI_IP_ADDRESS>

# you can run the following script:

import cv2
import subprocess
import numpy as np

WIDTH = 1280
HEIGHT = 720

FFMPEG_CMD = [
    "ffmpeg",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-analyzeduration", "0",
    "-probesize", "32",
    "-i", "tcp://10.207.55.64:5001?listen=0",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]



p = subprocess.Popen(
    FFMPEG_CMD,
    stdout=subprocess.PIPE,
    bufsize=WIDTH * HEIGHT * 3
)

buffer_size = WIDTH * HEIGHT * 3

while True:
    raw = p.stdout.read(buffer_size)
    if len(raw) != buffer_size:
        continue

    frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))

    cv2.imshow("Pi Stream", frame)
    if cv2.waitKey(1) == ord("q"):
        break

p.terminate()
cv2.destroyAllWindows()
