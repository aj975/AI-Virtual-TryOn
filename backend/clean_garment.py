
import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

source = BASE / "garments" / "blue_tshirt.webp"
destination = BASE / "garments" / "blue_tshirt_clean.png"

image = cv2.imread(str(source))

if image is None:
    raise FileNotFoundError(source)

# Identify the checkerboard background by its
# low saturation and light gray/white colors.
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

background = cv2.inRange(
    hsv,
    np.array([0, 0, 175]),
    np.array([179, 45, 255])
)

# Keep the largest foreground object (the shirt).
foreground = cv2.bitwise_not(background)

count, labels, stats, _ = cv2.connectedComponentsWithStats(
    foreground
)

mask = np.zeros(image.shape[:2], dtype=np.uint8)

if count > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask[labels == largest] = 255

# Smooth the garment boundary.
mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    np.ones((5, 5), np.uint8)
)

mask = cv2.GaussianBlur(mask, (3, 3), 0)

# Place shirt on a white background.
alpha = mask.astype(np.float32) / 255.0

white = np.full_like(image, 255)

result = (
    image.astype(np.float32) * alpha[:, :, None]
    + white.astype(np.float32) * (1 - alpha[:, :, None])
).astype(np.uint8)

cv2.imwrite(str(destination), result)

print("Clean garment saved:", destination)
