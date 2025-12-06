import cv2
import subprocess
import numpy as np

WIDTH = 320
HEIGHT = 240

# YUV420p = Y plane (WIDTH*HEIGHT bytes) + U+V downsampled planes
# Total size = WIDTH * HEIGHT * 3/2
FRAME_SIZE = int(WIDTH * HEIGHT * 3 // 2)

FFMPEG_CMD = [
    "ffmpeg",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-analyzeduration", "0",
    "-probesize", "32",
    "-i", "tcp://10.207.70.178:5001?listen=0",
    "-f", "rawvideo",
    "-pix_fmt", "yuv420p",
    "-"
]


p = subprocess.Popen(
    FFMPEG_CMD,
    stdout=subprocess.PIPE,
    bufsize=FRAME_SIZE
)

while True:
    raw = p.stdout.read(FRAME_SIZE)
    if len(raw) != FRAME_SIZE:
        continue  # incomplete frame, skip

    # Interpret buffer as YUV420p
    yuv = np.frombuffer(raw, dtype=np.uint8)

    # Extract luminance (Y) plane only
    Y = yuv[0 : WIDTH * HEIGHT].reshape((HEIGHT, WIDTH))

    # Display grayscale Y channel (for debugging)
    cv2.imshow("Y-Channel Stream", Y)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

p.terminate()
cv2.destroyAllWindows()
