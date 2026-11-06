
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V5.5 - YELLOW EDGE CLEANUP
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

original = cv2.imread(
    str(OUT / "v54_best_composite_backup.png")
)

ai = cv2.imread(
    str(OUT / "best_fitting_backup.png")
)

shirt_mask = cv2.imread(
    str(OUT / "v54_shirt_mask.png"),
    cv2.IMREAD_GRAYSCALE
)

if any(x is None for x in [original, ai, shirt_mask]):
    raise FileNotFoundError("Missing V5.4 backup, AI image, or shirt mask.")

h, w = original.shape[:2]
ai = cv2.resize(ai, (w, h), interpolation=cv2.INTER_LINEAR)
shirt_mask = cv2.resize(
    shirt_mask, (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# DETECT YELLOW PIXELS
# ==========================================

hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

yellow = cv2.inRange(
    hsv,
    np.array([15, 45, 45], dtype=np.uint8),
    np.array([45, 255, 255], dtype=np.uint8)
)

# ==========================================
# LIMIT CLEANUP TO SHIRT EDGES
# ==========================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (13, 13)
)

near_shirt = cv2.dilate(shirt_mask, kernel)

candidate = cv2.bitwise_and(yellow, near_shirt)

# Restrict to upper-body region.
upper_body = np.zeros((h, w), dtype=np.uint8)
upper_body[int(h * 0.25):int(h * 0.94), :] = 255

candidate = cv2.bitwise_and(candidate, upper_body)

# Exclude areas far outside the original shirt.
# This limits accidental replacement of skin.
distance = cv2.distanceTransform(
    cv2.bitwise_not(shirt_mask),
    cv2.DIST_L2,
    3
)

candidate[distance > 5] = 0

# ==========================================
# REPLACE CANDIDATE PIXELS WITH AI IMAGE
# ==========================================

alpha = cv2.GaussianBlur(
    candidate.astype(np.float32) / 255.0,
    (5, 5),
    0
)

alpha = alpha[:, :, None]

result = (
    ai.astype(np.float32) * alpha
    + original.astype(np.float32) * (1 - alpha)
)

result = np.clip(result, 0, 255).astype(np.uint8)

# ==========================================
# SAVE
# ==========================================

mask_path = OUT / "v55_yellow_cleanup_mask.png"
result_path = OUT / "v55_edge_cleanup_result.png"
comparison_path = OUT / "v55_before_after.png"

cv2.imwrite(str(mask_path), candidate)
cv2.imwrite(str(result_path), result)
cv2.imwrite(
    str(comparison_path),
    np.hstack([original, result])
)

print("\nVIRTUALFIT V5.5 COMPLETE!")
print("Candidate pixels:", cv2.countNonZero(candidate))
print("Saved:", result_path)
print("Saved:", comparison_path)
print("V5.4 backup unchanged.")
