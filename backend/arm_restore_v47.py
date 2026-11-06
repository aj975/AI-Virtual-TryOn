
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V4.7 - ARM RESTORATION PREVIEW
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original_path = OUT / f"capture_{SESSION}.png"
ai_path = OUT / "v47_before_arm_restoration.png"

original = cv2.imread(str(original_path))
ai = cv2.imread(str(ai_path))

if original is None or ai is None:
    raise FileNotFoundError(
        "Original photo or V4.7 backup not found."
    )

h, w = original.shape[:2]
ai = cv2.resize(ai, (w, h))

# Load the person masks created in V4.5
original_person = cv2.imread(
    str(OUT / "v45_original_person_mask.png"),
    cv2.IMREAD_GRAYSCALE
)

ai_person = cv2.imread(
    str(OUT / "v45_ai_person_mask.png"),
    cv2.IMREAD_GRAYSCALE
)

if original_person is None or ai_person is None:
    raise FileNotFoundError(
        "V4.5 person masks missing. Run V4.5 first."
    )

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
# Based on V4.3 shoulder/elbow landmarks.
# Image-left and image-right refer to the
# viewer's perspective.
# ==========================================

# Start conservatively below the sleeve hems.
# Restrict work to exposed arms only.
left_polygon = np.array([
    [115, 310],
    [205, 310],
    [228, 345],
    [230, h - 1],
    [115, h - 1]
], dtype=np.int32)

right_polygon = np.array([
    [470, 310],
    [565, 310],
    [590, h - 1],
    [455, h - 1],
    [465, 345]
], dtype=np.int32)

region = np.zeros((h, w), dtype=np.uint8)

cv2.fillPoly(
    region,
    [left_polygon, right_polygon],
    255
)

# Only copy pixels classified as person in the
# original image, inside the arm regions.
arm_mask = cv2.bitwise_and(
    original_person,
    region
)

# Clean small holes in the original silhouette.
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

arm_mask = cv2.morphologyEx(
    arm_mask,
    cv2.MORPH_CLOSE,
    kernel
)

# Keep only the exposed-arm area; use a small
# feather for smoother transitions.
arm_mask = cv2.GaussianBlur(
    arm_mask,
    (9, 9),
    0
)

# ==========================================
# COMPOSITE
# ==========================================

alpha = arm_mask.astype(np.float32) / 255.0
alpha = alpha[:, :, None]

result = (
    original.astype(np.float32) * alpha
    + ai.astype(np.float32) * (1.0 - alpha)
)

result = np.clip(result, 0, 255).astype(np.uint8)

# ==========================================
# SAVE RESULTS
# ==========================================

mask_path = OUT / "v47_arm_restoration_mask.png"
result_path = OUT / "v47_arm_restored_preview.png"
comparison_path = OUT / "v47_before_after.png"

cv2.imwrite(str(mask_path), arm_mask)
cv2.imwrite(str(result_path), result)

comparison = np.hstack([ai, result])
cv2.imwrite(str(comparison_path), comparison)

print("\nVIRTUALFIT V4.7 COMPLETE!")
print("Saved:", result_path)
print("Comparison:", comparison_path)
print("Mask:", mask_path)
print("Original and backup files unchanged.")

