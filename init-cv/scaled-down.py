import cv2
import subprocess
import numpy as np

WIDTH = 640
HEIGHT = 360

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
