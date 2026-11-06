
import cv2
import numpy as np
from pathlib import Path

# VIRTUALFIT V5.6 - NECKLINE REPAIR
# Uses V5.4 as the baseline.
# Does not modify existing results.

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

image = cv2.imread(
    str(OUT / "v54_best_composite_backup.png")
)

if image is None:
    raise FileNotFoundError("V5.4 backup not found.")

h, w = image.shape[:2]

# The neckline is near the center of the upper body.
# Restrict the repair to a small collar region.
x1 = int(w * 0.40)
x2 = int(w * 0.60)
y1 = int(h * 0.32)
y2 = int(h * 0.44)

region = np.zeros((h, w), dtype=np.uint8)
region[y1:y2, x1:x2] = 255

# Detect yellow pixels in HSV color space.
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

yellow = cv2.inRange(
    hsv,
    np.array([17, 55, 65], dtype=np.uint8),
    np.array([43, 255, 255], dtype=np.uint8)
)

mask = cv2.bitwise_and(yellow, region)

# Slightly expand selected pixels to cover yellow fringes.
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)
mask = cv2.dilate(mask, kernel, iterations=1)

# Repair only selected pixels.
result = cv2.inpaint(
    image,
    mask,
    3,
    cv2.INPAINT_TELEA
)

mask_path = OUT / "v56_neckline_mask.png"
result_path = OUT / "v56_neckline_result.png"
comparison_path = OUT / "v56_before_after.png"

cv2.imwrite(str(mask_path), mask)
cv2.imwrite(str(result_path), result)
cv2.imwrite(
    str(comparison_path),
    np.hstack([image, result])
)

print("\nVIRTUALFIT V5.6 COMPLETE!")
print("Selected pixels:", cv2.countNonZero(mask))
print("Saved:", result_path)
print("V5.4 backup unchanged.")
