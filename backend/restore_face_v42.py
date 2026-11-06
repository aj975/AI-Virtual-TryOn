
import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original = cv2.imread(
    str(OUT / f"capture_{SESSION}.png")
)

ai = cv2.imread(
    str(OUT / "best_fitting_backup.png")
)

if original is None or ai is None:
    raise FileNotFoundError("Original or AI image not found.")

# Match image dimensions
h, w = original.shape[:2]
ai = cv2.resize(ai, (w, h))

# Detect faces in both images
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4,
        minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])

original_face = detect_face(original)
ai_face = detect_face(ai)

if original_face is None or ai_face is None:
    raise RuntimeError(
        "Face not detected in both images. "
        "Original AI result remains unchanged."
    )

# Align original face center with AI face center
ox, oy, ow, oh = original_face
ax, ay, aw, ah = ai_face

dx = (ax + aw / 2) - (ox + ow / 2)
dy = (ay + ah / 2) - (oy + oh / 2)

matrix = np.float32([
    [1, 0, dx],
    [0, 1, dy]
])

aligned = cv2.warpAffine(
    original, matrix, (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_REFLECT
)

# Small elliptical mask around face only.
# Avoid hair, neck, and T-shirt neckline.
mask = np.zeros((h, w), dtype=np.uint8)

cx = int(ax + aw / 2)
cy = int(ay + ah / 2)
rx = int(aw * 0.43)
ry = int(ah * 0.48)

cv2.ellipse(
    mask, (cx, cy), (rx, ry),
    0, 0, 360, 255, -1
)

# Feather face edges
mask = cv2.GaussianBlur(mask, (31, 31), 0)

alpha = mask.astype(np.float32)[:, :, None] / 255.0

final = (
    aligned.astype(np.float32) * alpha
    + ai.astype(np.float32) * (1 - alpha)
)

final = np.clip(final, 0, 255).astype(np.uint8)

output = OUT / "v42_face_restored.png"

if not cv2.imwrite(str(output), final):
    raise RuntimeError("Could not save result.")

print("VIRTUALFIT V4.2 COMPLETE!")
print("Saved:", output)
print("Original face:", original_face)
print("AI face:", ai_face)
print("Blue shirt preserved outside face mask.")
