
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V3.8 - TARGETED WAIST FIX
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "8658aeb79b8c4897903ad0094bf8a262"

original = cv2.imread(
    str(OUT / f"capture_{SESSION}.png")
)

ai = cv2.imread(
    str(OUT / f"web_tryon_{SESSION}.png")
)

v36 = cv2.imread(
    str(OUT / "v36_smooth_fitted_result.png")
)

shirt_mask = cv2.imread(
    str(OUT / "v3_final_mask_v33.png"), 0
)

arms_mask = cv2.imread(
    str(OUT / "v3_arms_mask.png"), 0
)

images = [original, ai, v36, shirt_mask, arms_mask]

if any(img is None for img in images):
    raise FileNotFoundError("Missing input image.")

h, w = original.shape[:2]

ai = cv2.resize(ai, (w, h))
v36 = cv2.resize(v36, (w, h))

shirt_mask = cv2.resize(
    shirt_mask, (w, h),
    interpolation=cv2.INTER_NEAREST
)

arms_mask = cv2.resize(
    arms_mask, (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# FIND WAIST AREA
# ==========================================

shirt_binary = shirt_mask > 127

ys = np.where(shirt_binary)[0]
top = int(ys.min())
bottom = int(ys.max())

# Focus on lower 45% of the shirt
waist_start = int(top + 0.55 * (bottom - top))

waist_region = np.zeros((h, w), dtype=np.uint8)
waist_region[waist_start:bottom + 1, :] = 255

# ==========================================
# DETECT REMAINING YELLOW FABRIC
# ==========================================

hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

yellow = cv2.inRange(
    hsv,
    np.array([15, 45, 45], dtype=np.uint8),
    np.array([45, 255, 255], dtype=np.uint8)
)

# Extend search around shirt edges
expanded_shirt = cv2.dilate(
    shirt_mask,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (61, 61)
    )
)

candidate = cv2.bitwise_and(yellow, expanded_shirt)
candidate = cv2.bitwise_and(candidate, waist_region)

# Only cover pixels outside existing shirt
candidate = cv2.bitwise_and(
    candidate,
    cv2.bitwise_not(shirt_mask)
)

# Protect arms, with a smaller margin
protected_arms = cv2.dilate(
    arms_mask,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
)

candidate = cv2.bitwise_and(
    candidate,
    cv2.bitwise_not(protected_arms)
)

# ==========================================
# SMOOTH PATCH MASK
# ==========================================

candidate = cv2.morphologyEx(
    candidate,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
)

soft_mask = cv2.GaussianBlur(
    candidate, (9, 9), 0
)

# ==========================================
# BLEND AI FABRIC INTO V3.6
# ==========================================

alpha = (
    soft_mask.astype(np.float32) / 255.0
)[:, :, None]

final = (
    ai.astype(np.float32) * alpha
    + v36.astype(np.float32) * (1 - alpha)
)

final = np.clip(final, 0, 255).astype(np.uint8)

# ==========================================
# SAVE RESULTS
# ==========================================

result_path = OUT / "v38_waist_fixed.png"
mask_path = OUT / "v38_waist_mask.png"

cv2.imwrite(str(result_path), final)
cv2.imwrite(str(mask_path), candidate)

print("VIRTUALFIT V3.8 COMPLETE!")
print("Result:", result_path)
print("Mask:", mask_path)
print("Selected pixels:", cv2.countNonZero(candidate))

cv2.imshow("VIRTUALFIT V3.8", final)

print("Press Q to close.")

while True:
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
