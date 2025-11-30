import cv2
import subprocess
import numpy as np

# Match rpicam-vid settings
WIDTH = 1280
HEIGHT = 720

PI_IP = "10.207.55.64"
PORT = 5001

FFMPEG_CMD = [
    "ffmpeg",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-analyzeduration", "0",
    "-probesize", "32",
    "-i", f"tcp://{PI_IP}:{PORT}?listen=0",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

# Regions of interest: (x, y, w, h)
ROIS = {
    "top_left": (
        int(WIDTH * 0.05),
        int(HEIGHT * 0.05),
        100,
        100
    ),
    "center": (
        int(WIDTH * 0.5) - 50,
        int(HEIGHT * 0.5) - 50,
        100,
        100
    ),
    "bottom_right": (
        int(WIDTH * 0.95) - 100,
        int(HEIGHT * 0.95) - 100,
        100,
        100
    ),
}

def main():
    p = subprocess.Popen(
        FFMPEG_CMD,
        stdout=subprocess.PIPE,
        bufsize=WIDTH * HEIGHT * 3
    )

    buffer_size = WIDTH * HEIGHT * 3

    try:
        while True:
            raw = p.stdout.read(buffer_size)
            if len(raw) != buffer_size:
                # Incomplete frame; skip or break
                continue

            # frombuffer -> reshape -> copy to make it writable
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (HEIGHT, WIDTH, 3)
            ).copy()

            # Grayscale for luminance and edges
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Edges (Canny)
            edges = cv2.Canny(gray, 100, 200)

            # Luminance in each ROI
            luminances = {}
            for name, (x, y, w, h) in ROIS.items():
                roi = gray[y:y + h, x:x + w]
                if roi.size > 0:
                    luminances[name] = float(np.mean(roi))
                else:
                    luminances[name] = float("nan")

                # Draw ROI rectangles on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    name,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Print luminance values
            print("Luminance (0–255):",
                  {k: round(v, 1) for k, v in luminances.items()})

            # Show original + edges
            cv2.imshow("Pi Stream (ROIs)", frame)
            cv2.imshow("Edges", edges)

            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        p.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
