import cv2
import numpy as np
import subprocess

WIDTH = 640
HEIGHT = 360

FFMPEG_CMD = [
    "ffmpeg",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-fflags", "discardcorrupt",
    "-analyzeduration", "0",
    "-probesize", "32",
    "-i", "udp://0.0.0.0:5001?fifo_size=500000&overrun_nonfatal=1",
    "-f", "rawvideo",
    "-pix_fmt", "gray",
    "-"
]

p = subprocess.Popen(
    FFMPEG_CMD,
    stdout=subprocess.PIPE,
    bufsize=WIDTH * HEIGHT
)

buffer_size = WIDTH * HEIGHT

while True:
    raw = p.stdout.read(buffer_size)
    if len(raw) != buffer_size:
        continue

    frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH))

    cv2.imshow("Low-Latency RTP Stream", frame)
    if cv2.waitKey(1) == ord("q"):
        break

p.terminate()
cv2.destroyAllWindows()
