
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V4.0 - TARGETED WAIST REPAIR
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "8658aeb79b8c4897903ad0094bf8a262"

original_path = OUT / f"capture_{SESSION}.png"
result_path = OUT / "v36_smooth_fitted_result.png"
shirt_path = OUT / "v3_final_mask_v33.png"
arms_path = OUT / "v3_arms_mask.png"

original = cv2.imread(str(original_path))
result = cv2.imread(str(result_path))
shirt = cv2.imread(str(shirt_path), 0)
arms = cv2.imread(str(arms_path), 0)

if any(x is None for x in [original, result, shirt, arms]):
    raise FileNotFoundError("One or more input files are missing.")

h, w = result.shape[:2]

original = cv2.resize(original, (w, h))
shirt = cv2.resize(shirt, (w, h))
arms = cv2.resize(arms, (w, h))

# ==========================================
# DETECT YELLOW IN FINAL RESULT
# ==========================================

hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

yellow = cv2.inRange(
    hsv,
    np.array([15, 45, 45], dtype=np.uint8),
    np.array([45, 255, 255], dtype=np.uint8)
)

# ==========================================
# FIND SHIRT POSITION
# ==========================================

ys = np.where(shirt > 127)[0]

if len(ys) == 0:
    raise RuntimeError("Shirt mask is empty.")

top = int(ys.min())
bottom = int(ys.max())

waist_start = int(top + 0.55 * (bottom - top))

# ==========================================
# DEFINE TARGET AREA
# ==========================================

# Focus on left side of image near waist
# This is specific to our current test photo.

roi = np.zeros((h, w), dtype=np.uint8)

roi[
    waist_start:bottom + 1,
    int(w * 0.28):int(w * 0.50)
] = 255

# ==========================================
# BUILD REPAIR MASK
# ==========================================

mask = cv2.bitwise_and(yellow, roi)

# Expand detected yellow slightly
mask = cv2.dilate(
    mask,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)
    )
)

# Protect arms
protected_arms = cv2.dilate(
    arms,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
)

mask[protected_arms > 127] = 0

# ==========================================
# INPAINT
# ==========================================

repaired = cv2.inpaint(
    result,
    mask,
    5,
    cv2.INPAINT_TELEA
)

# ==========================================
# SAVE RESULTS
# ==========================================

output_path = OUT / "v40_waist_repaired.png"
mask_path = OUT / "v40_repair_mask.png"

cv2.imwrite(str(output_path), repaired)
cv2.imwrite(str(mask_path), mask)

print("VIRTUALFIT V4.0 COMPLETE!")
print("Repair pixels:", cv2.countNonZero(mask))
print("Saved:", output_path)

cv2.imshow("VIRTUALFIT V4.0", repaired)

print("Press any key in the image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
