import cv2
import numpy as np
from pathlib import Path

base = Path(__file__).resolve().parent.parent
out = base / "outputs"

image = cv2.imread(str(base / "assets" / "person.jpeg"))
shirt = cv2.imread(str(out / "shirt_mask.png"), 0)
arms = cv2.imread(str(out / "arms_mask.png"), 0)

overlay = image.copy()

overlay[shirt > 127] = (
    overlay[shirt > 127] * 0.45 +
    np.array([255, 0, 0]) * 0.55
).astype(np.uint8)

overlay[arms > 127] = (
    overlay[arms > 127] * 0.45 +
    np.array([0, 0, 255]) * 0.55
).astype(np.uint8)

cv2.imwrite(str(out / "segmentation_overlay.png"), overlay)

print("Colored segmentation preview saved!")
