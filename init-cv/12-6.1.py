# -------------------------------------------------------------
# RUN THIS COMMAND ON THE RASPBERRY PI OVER SSH:
'''
rpicam-vid \
    --codec h264 \
    --profile baseline \
    --level 4.2 \
    --width 320 \
    --height 240 \
    --framerate 15 \
    --bitrate 1500000 \
    --intra 15 \
    --inline \
    --low-latency \
    --libav-video-codec-opts "tune=zerolatency;preset=ultrafast" \
    --sharpness 1.2 \
    --contrast 1.1 \
    --denoise off \
    --exposure sport \
    --shutter 8000 \
    --awb custom \
    --awbgains 1.0,1.0 \
    --timeout 0 \
    --buffer-count 2 \
    --flush \
    -n \
    --listen \
    -o tcp://0.0.0.0:5001
'''
# -------------------------------------------------------------


import cv2
import subprocess
import numpy as np

WIDTH = 320
HEIGHT = 240
PIXEL_BYTES = 3

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

    cv2.imshow("Pi Stream (Stable 15fps H.264)", gray)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

p.terminate()
cv2.destroyAllWindows()
