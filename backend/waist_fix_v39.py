
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V3.9 - TARGETED WAIST REPAIR
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "8658aeb79b8c4897903ad0094bf8a262"

def load(name, grayscale=False):
    path = OUT / name
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE if grayscale
        else cv2.IMREAD_COLOR
    )
    if image is None:
        raise FileNotFoundError(str(path))
    return image

original = load(f"capture_{SESSION}.png")
ai = load(f"web_tryon_{SESSION}.png")
v36 = load("v36_smooth_fitted_result.png")
shirt = load("v3_final_mask_v33.png", True)
arms = load("v3_arms_mask.png", True)

h, w = original.shape[:2]

ai = cv2.resize(ai, (w, h))
v36 = cv2.resize(v36, (w, h))

shirt = cv2.resize(
    shirt, (w, h),
    interpolation=cv2.INTER_NEAREST
)

arms = cv2.resize(
    arms, (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# IDENTIFY LOWER SHIRT
# ==========================================

binary = shirt > 127
ys = np.where(binary)[0]

if len(ys) == 0:
    raise RuntimeError("Shirt mask is empty")

top = int(ys.min())
bottom = int(ys.max())

waist_start = int(top + 0.58 * (bottom - top))

# ==========================================
# FIND YELLOW FABRIC
# ==========================================

hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

yellow = cv2.inRange(
    hsv,
    np.array([15, 45, 45], dtype=np.uint8),
    np.array([45, 255, 255], dtype=np.uint8)
)

# Search near the original shirt boundary
near_shirt = cv2.dilate(
    shirt,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (81, 81)
    )
)

candidate = cv2.bitwise_and(yellow, near_shirt)

# Restrict to lower shirt
candidate[:waist_start, :] = 0
candidate[bottom + 1:, :] = 0

# Work only outside existing mask
candidate[shirt > 127] = 0

# ==========================================
# PROTECT ARMS
# ==========================================

protected = cv2.dilate(
    arms,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)
    )
)

candidate[protected > 127] = 0

# ==========================================
# EXPAND ONLY DETECTED PATCHES
# ==========================================

expanded = cv2.dilate(
    candidate,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 15)
    )
)

# Keep the repair near yellow fabric
yellow_nearby = cv2.dilate(
    yellow,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
)

expanded = cv2.bitwise_and(
    expanded, yellow_nearby
)

expanded[protected > 127] = 0
expanded[:waist_start, :] = 0
expanded[bottom + 1:, :] = 0

# ==========================================
# CHECK AI REPLACEMENT IS BLUE
# ==========================================

ai_hsv = cv2.cvtColor(ai, cv2.COLOR_BGR2HSV)

blue = cv2.inRange(
    ai_hsv,
    np.array([95, 30, 35], dtype=np.uint8),
    np.array([135, 255, 255], dtype=np.uint8)
)

expanded = cv2.bitwise_and(expanded, blue)

# ==========================================
# BLEND PATCH
# ==========================================

soft = cv2.GaussianBlur(
    expanded, (9, 9), 0
)

alpha = soft.astype(np.float32)[:, :, None] / 255.0

final = (
    ai.astype(np.float32) * alpha
    + v36.astype(np.float32) * (1 - alpha)
)

final = np.clip(final, 0, 255).astype(np.uint8)

# ==========================================
# SAVE OUTPUTS
# ==========================================

result_path = OUT / "v39_waist_fixed.png"
mask_path = OUT / "v39_waist_mask.png"

if not cv2.imwrite(str(result_path), final):
    raise RuntimeError("Could not save result")

cv2.imwrite(str(mask_path), expanded)

print("VIRTUALFIT V3.9 COMPLETE!")
print("Selected pixels:", cv2.countNonZero(expanded))
print("Result:", result_path)

cv2.imshow("VIRTUALFIT V3.9", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
