
import cv2
import numpy as np
from pathlib import Path

# VIRTUALFIT V5.3 - SKIN MASK DIAGNOSTIC
# No image restoration performed.

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original = cv2.imread(
    str(OUT / f"capture_{SESSION}.png")
)

if original is None:
    raise FileNotFoundError("Original capture not found.")

h, w = original.shape[:2]

# Skin-color candidates in YCrCb space.
ycrcb = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)

skin = cv2.inRange(
    ycrcb,
    np.array([0, 130, 70], dtype=np.uint8),
    np.array([255, 185, 140], dtype=np.uint8)
)

# Restrict to exposed-arm areas.
region = np.zeros((h, w), dtype=np.uint8)

left_polygon = np.array([
    [175, 310],
    [250, 310],
    [255, h - 1],
    [140, h - 1]
], dtype=np.int32)

right_polygon = np.array([
    [460, 310],
    [560, 310],
    [610, h - 1],
    [450, h - 1]
], dtype=np.int32)

cv2.fillPoly(
    region,
    [left_polygon, right_polygon],
    255
)

skin = cv2.bitwise_and(skin, region)

# Remove isolated noise.
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

skin = cv2.morphologyEx(
    skin, cv2.MORPH_OPEN, kernel
)

# Diagnostic overlay.
preview = original.copy()

overlay = original.copy()
overlay[skin > 0] = (255, 0, 255)

preview = cv2.addWeighted(
    original, 0.55,
    overlay, 0.45,
    0
)

mask_path = OUT / "v53_skin_mask.png"
preview_path = OUT / "v53_skin_preview.png"

cv2.imwrite(str(mask_path), skin)
cv2.imwrite(str(preview_path), preview)

print("\nVIRTUALFIT V5.3 COMPLETE!")
print("Skin candidate pixels:", cv2.countNonZero(skin))
print("Saved:", mask_path)
print("Saved:", preview_path)
print("Blue T-shirt unchanged.")
