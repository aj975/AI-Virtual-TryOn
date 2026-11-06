
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V4.8 - PRECISION ARM MASK PREVIEW
# NO IMAGE RESTORATION IN THIS VERSION
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original = cv2.imread(str(OUT / f"capture_{SESSION}.png"))
ai = cv2.imread(str(OUT / "best_fitting_backup.png"))

original_person = cv2.imread(
    str(OUT / "v45_original_person_mask.png"),
    cv2.IMREAD_GRAYSCALE
)
ai_person = cv2.imread(
    str(OUT / "v45_ai_person_mask.png"),
    cv2.IMREAD_GRAYSCALE
)

if any(x is None for x in [
    original, ai, original_person, ai_person
]):
    raise FileNotFoundError(
        "Missing original, AI, or V4.5 mask file."
    )

h, w = original.shape[:2]

ai = cv2.resize(ai, (w, h))

original_person = cv2.resize(
    original_person, (w, h),
    interpolation=cv2.INTER_NEAREST
)

ai_person = cv2.resize(
    ai_person, (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# ARM REGIONS
# Narrow corridors around the exposed arms.
# Coordinates are based on the V4.3 landmarks
# for the current saved photograph.
# ==========================================

# Image-left arm
left_region = np.array([
    [185, 335],
    [230, 335],
    [255, h - 1],
    [170, h - 1]
], dtype=np.int32)

# Image-right arm
right_region = np.array([
    [465, 330],
    [535, 330],
    [570, h - 1],
    [465, h - 1]
], dtype=np.int32)

region = np.zeros((h, w), dtype=np.uint8)

cv2.fillPoly(
    region,
    [left_region, right_region],
    255
)

# Intersect the original person silhouette with
# the allowed arm regions.
candidate = cv2.bitwise_and(
    original_person,
    region
)

# Conservative erosion to stay inside the
# original arm boundaries.
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (7, 7)
)

candidate = cv2.erode(
    candidate,
    kernel,
    iterations=1
)

# Keep only the mask regions that also fall
# inside the AI person silhouette.
# This is deliberately conservative.
candidate = cv2.bitwise_and(
    candidate,
    ai_person
)

# ==========================================
# VISUALIZATION ONLY
# ==========================================

preview = ai.copy()

# Bright magenta indicates proposed replacement
# pixels. No pixels are actually replaced.
overlay = preview.copy()
overlay[candidate > 0] = (255, 0, 255)

preview = cv2.addWeighted(
    preview, 0.55,
    overlay, 0.45,
    0
)

mask_path = OUT / "v48_arm_candidate_mask.png"
preview_path = OUT / "v48_arm_mask_preview.png"

cv2.imwrite(str(mask_path), candidate)
cv2.imwrite(str(preview_path), preview)

print("\nVIRTUALFIT V4.8 COMPLETE!")
print("Candidate pixels:", cv2.countNonZero(candidate))
print("Mask:", mask_path)
print("Preview:", preview_path)
print("No photograph was modified.")
