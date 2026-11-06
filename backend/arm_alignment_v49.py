
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V4.9 - ARM ALIGNMENT ANALYSIS
# Diagnostic only: does NOT modify the shirt
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

if any(item is None for item in [
    original, ai, original_person, ai_person
]):
    raise FileNotFoundError(
        "Missing original photo, AI photo, or V4.5 person masks."
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
# ARM-ONLY SEARCH REGIONS
# Restrict comparisons below the sleeve hems
# to avoid including the blue T-shirt.
# ==========================================

region = np.zeros((h, w), dtype=np.uint8)

left_region = np.array([
    [175, 335],
    [245, 335],
    [255, h - 1],
    [150, h - 1]
], dtype=np.int32)

right_region = np.array([
    [460, 330],
    [550, 330],
    [585, h - 1],
    [455, h - 1]
], dtype=np.int32)

cv2.fillPoly(
    region,
    [left_region, right_region],
    255
)

# ==========================================
# EXTRACT PERSON SILHOUETTES
# ==========================================

original_arms = cv2.bitwise_and(
    original_person, region
)

ai_arms = cv2.bitwise_and(
    ai_person, region
)

# Find outlines, not filled regions.
original_edges = cv2.morphologyEx(
    original_arms,
    cv2.MORPH_GRADIENT,
    np.ones((3, 3), dtype=np.uint8)
)

ai_edges = cv2.morphologyEx(
    ai_arms,
    cv2.MORPH_GRADIENT,
    np.ones((3, 3), dtype=np.uint8)
)

# ==========================================
# DRAW OUTLINES ON AI PHOTO
# OpenCV uses BGR color order.
# Original = GREEN
# AI = RED
# ==========================================

preview = ai.copy()

preview[original_edges > 0] = (0, 255, 0)
preview[ai_edges > 0] = (0, 0, 255)

# Mark the approximate elbow positions.
# These were measured in V4.3.
original_elbows = [
    (210, 389),
    (505, 382)
]

ai_elbows = [
    (220, 372),
    (509, 370)
]

for x, y in original_elbows:
    cv2.circle(
        preview, (x, y), 5,
        (0, 255, 0), -1
    )

for x, y in ai_elbows:
    cv2.circle(
        preview, (x, y), 5,
        (0, 0, 255), -1
    )

# ==========================================
# SAVE DIAGNOSTICS
# ==========================================

preview_path = OUT / "v49_arm_alignment.png"
original_mask_path = OUT / "v49_original_arm_regions.png"
ai_mask_path = OUT / "v49_ai_arm_regions.png"

cv2.imwrite(str(preview_path), preview)
cv2.imwrite(str(original_mask_path), original_arms)
cv2.imwrite(str(ai_mask_path), ai_arms)

print("\nVIRTUALFIT V4.9 COMPLETE!")
print("GREEN outlines = Original body")
print("RED outlines = AI-generated body")
print("Saved:", preview_path)
print("No changes made to your T-shirt or website.")
