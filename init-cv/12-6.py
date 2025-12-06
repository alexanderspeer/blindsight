#RELIABLE STREAMING
#STUTTERS WHENEVER CAMERA REALLY MOVES, BUT OTHER THAN THAT IS PRETTY GOOD



'''
rpicam-vid \
    --codec h264 \
    --profile baseline \
    --level 4.2 \
    --width 640 \
    --height 480 \
    --framerate 30 \
    --bitrate 2000000 \
    --intra 30 \
    --inline \
    --sharpness 1.2 \
    --denoise off \
    --timeout 0 \
    --buffer-count 2 \
    --flush \
    -n \
    --listen \
    -o tcp://0.0.0.0:5001
'''
# -------------------------------------------------------------
# This configuration is the ONLY format fully supported by IMX708
# for continuous, stable, low-latency streaming over TCP.
# -------------------------------------------------------------


import cv2
import subprocess
import numpy as np

WIDTH = 640
HEIGHT = 480
PIXEL_BYTES = 3   # bgr24

FFMPEG_CMD = [
    "ffmpeg",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-analyzeduration", "0",
    "-probesize", "32",
    "-i", "tcp://10.207.70.178:5001?listen=0",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

p = subprocess.Popen(
    FFMPEG_CMD,
    stdout=subprocess.PIPE,
    bufsize=WIDTH * HEIGHT * PIXEL_BYTES
)

frame_size = WIDTH * HEIGHT * PIXEL_BYTES

while True:
    raw = p.stdout.read(frame_size)
    if len(raw) != frame_size:
        continue

    frame = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Pi Stream (Stable H.264)", gray)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

p.terminate()
cv2.destroyAllWindows()
