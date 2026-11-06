import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

person = cv2.imread(str(BASE / "assets/person.jpeg"))
garment = cv2.imread(
    str(BASE / "garments/blue_tshirt_transparent.png"),
    cv2.IMREAD_UNCHANGED
)
shirt_mask = cv2.imread(str(OUT / "shirt_mask.png"), 0)
arms_mask = cv2.imread(str(OUT / "arms_mask.png"), 0)

if any(x is None for x in [person, garment, shirt_mask, arms_mask]):
    raise FileNotFoundError("One or more input files are missing.")

h, w = person.shape[:2]

# Find existing shirt outline
contours, _ = cv2.findContours(
    shirt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
x, y, sw, sh = cv2.boundingRect(
    max(contours, key=cv2.contourArea)
)

# Resize garment to match existing shirt bounds
fitted = cv2.resize(
    garment, (sw, sh),
    interpolation=cv2.INTER_LINEAR
)

# Build full-size garment layer
layer = np.zeros((h, w, 4), dtype=np.uint8)
layer[y:y+sh, x:x+sw] = fitted

# Clean the existing shirt mask
kernel = np.ones((9, 9), np.uint8)
clean_mask = cv2.morphologyEx(
    shirt_mask, cv2.MORPH_CLOSE, kernel
)

# Slight expansion to cover edge gaps
clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

# Preserve exposed arms
clean_mask[arms_mask > 127] = 0

# Combine garment transparency with shirt shape
alpha = layer[:, :, 3].astype(np.float32) / 255.0
alpha *= clean_mask.astype(np.float32) / 255.0

# Smooth final edges
alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
alpha = alpha[:, :, None]

result = (
    person.astype(np.float32) * (1 - alpha)
    + layer[:, :, :3].astype(np.float32) * alpha
).astype(np.uint8)

path = OUT / "garment_fit_v2.png"
cv2.imwrite(str(path), result)

print("GARMENT FIT V2 COMPLETE!")
print("Saved:", path)
