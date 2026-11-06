
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V5.2 - ARM RESTORATION PREVIEW
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original = cv2.imread(str(OUT / f"capture_{SESSION}.png"))
ai = cv2.imread(str(OUT / "v52_original_best_backup.png"))

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
        "Missing input image or person mask."
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
# LANDMARKS FROM V5.1
# ==========================================

arms = [
    {
        "name": "image_left",
        "original_shoulder": (253, 237),
        "original_elbow": (210, 389),
        "ai_shoulder": (266, 228),
        "ai_elbow": (220, 372),
        "xrange": (120, 290),
    },
    {
        "name": "image_right",
        "original_shoulder": (466, 230),
        "original_elbow": (505, 382),
        "ai_shoulder": (463, 227),
        "ai_elbow": (509, 370),
        "xrange": (445, 610),
    }
]

# ==========================================
# CREATE WARPED ORIGINAL IMAGE
# Smooth displacement based on elbow offsets.
# ==========================================

yy, xx = np.indices((h, w), dtype=np.float32)

map_x = xx.copy()
map_y = yy.copy()

for arm in arms:
    os = arm["original_shoulder"]
    oe = arm["original_elbow"]
    ais = arm["ai_shoulder"]
    aie = arm["ai_elbow"]

    x0, x1 = arm["xrange"]

    # Displacement grows from shoulder to elbow.
    shoulder_y = float(ais[1])
    elbow_y = float(aie[1])

    progress = np.clip(
        (yy - shoulder_y) /
        max(elbow_y - shoulder_y, 1),
        0, 1
    )

    dx_shoulder = ais[0] - os[0]
    dy_shoulder = ais[1] - os[1]

    dx_elbow = aie[0] - oe[0]
    dy_elbow = aie[1] - oe[1]

    dx = (
        (1 - progress) * dx_shoulder
        + progress * dx_elbow
    )

    dy = (
        (1 - progress) * dy_shoulder
        + progress * dy_elbow
    )

    region = (xx >= x0) & (xx <= x1)

    map_x[region] = xx[region] - dx[region]
    map_y[region] = yy[region] - dy[region]

warped_original = cv2.remap(
    original,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_REFLECT
)

warped_person = cv2.remap(
    original_person,
    map_x,
    map_y,
    interpolation=cv2.INTER_NEAREST,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
)

# ==========================================
# ARM REGIONS BELOW THE SLEEVES
# ==========================================

arm_region = np.zeros((h, w), dtype=np.uint8)

cv2.fillPoly(
    arm_region,
    [
        np.array([
            [185, 335],
            [270, 335],
            [270, h - 1],
            [140, h - 1]
        ], dtype=np.int32),
        np.array([
            [460, 330],
            [560, 330],
            [610, h - 1],
            [450, h - 1]
        ], dtype=np.int32)
    ],
    255
)

# Keep only pixels that belong to the
# warped original person silhouette.
arm_mask = cv2.bitwise_and(
    warped_person,
    arm_region
)

_, arm_mask = cv2.threshold(
    arm_mask, 127, 255,
    cv2.THRESH_BINARY
)

# Remove tiny isolated regions.
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

arm_mask = cv2.morphologyEx(
    arm_mask,
    cv2.MORPH_OPEN,
    kernel
)

# Feather boundaries for smoother blending.
alpha = cv2.GaussianBlur(
    arm_mask.astype(np.float32) / 255.0,
    (9, 9),
    0
)

alpha = alpha[:, :, None]

# ==========================================
# BLEND ORIGINAL ARMS INTO AI RESULT
# ==========================================

result = (
    warped_original.astype(np.float32) * alpha
    + ai.astype(np.float32) * (1.0 - alpha)
)

result = np.clip(result, 0, 255).astype(np.uint8)

# ==========================================
# SAVE PREVIEWS
# ==========================================

cv2.imwrite(
    str(OUT / "v52_arm_restoration_mask.png"),
    arm_mask
)

cv2.imwrite(
    str(OUT / "v52_arm_restoration_preview.png"),
    result
)

comparison = np.hstack([ai, result])

cv2.imwrite(
    str(OUT / "v52_before_after.png"),
    comparison
)

print("\nVIRTUALFIT V5.2 COMPLETE!")
print("Saved arm mask and before/after preview.")
print("Original and best-fitting AI files unchanged.")
