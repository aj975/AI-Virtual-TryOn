
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V3.7 - SHIRT EDGE CLEANUP
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "8658aeb79b8c4897903ad0094bf8a262"

original_path = OUT / f"capture_{SESSION}.png"
ai_path = OUT / f"web_tryon_{SESSION}.png"
v36_path = OUT / "v36_smooth_fitted_result.png"
shirt_mask_path = OUT / "v3_final_mask_v33.png"
arms_mask_path = OUT / "v3_arms_mask.png"

original = cv2.imread(str(original_path))
ai = cv2.imread(str(ai_path))
v36 = cv2.imread(str(v36_path))
shirt_mask = cv2.imread(str(shirt_mask_path), 0)
arms_mask = cv2.imread(str(arms_mask_path), 0)

required = {
    "original": original,
    "AI result": ai,
    "V3.6 result": v36,
    "shirt mask": shirt_mask,
    "arms mask": arms_mask,
}

for name, img in required.items():
    if img is None:
        raise FileNotFoundError(f"Missing {name}")

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
# DETECT ORIGINAL YELLOW FABRIC
# ==========================================

hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

yellow_mask = cv2.inRange(
    hsv,
    np.array([15, 45, 45], dtype=np.uint8),
    np.array([45, 255, 255], dtype=np.uint8)
)

# ==========================================
# SEARCH ONLY NEAR EXISTING SHIRT
# ==========================================

near_shirt = cv2.dilate(
    shirt_mask,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (31, 31)
    )
)

candidate = cv2.bitwise_and(
    yellow_mask, near_shirt
)

# Exclude shirt region already covered by V3.6
candidate = cv2.bitwise_and(
    candidate,
    cv2.bitwise_not(shirt_mask)
)

# Protect arms with a safety margin
protected_arms = cv2.dilate(
    arms_mask,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 15)
    )
)

candidate = cv2.bitwise_and(
    candidate,
    cv2.bitwise_not(protected_arms)
)

# ==========================================
# KEEP SMALL EDGE REGIONS
# ==========================================

num_labels, labels, stats, _ = (
    cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )
)

clean_mask = np.zeros((h, w), dtype=np.uint8)

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]

    if 8 <= area <= 4000:
        clean_mask[labels == i] = 255

# Slightly soften edges
clean_mask = cv2.GaussianBlur(
    clean_mask, (7, 7), 0
)

# ==========================================
# BLEND BLUE FABRIC INTO PATCHES
# ==========================================

alpha = (
    clean_mask.astype(np.float32) / 255.0
)[:, :, None]

final = (
    ai.astype(np.float32) * alpha
    + v36.astype(np.float32) * (1 - alpha)
)

final = np.clip(final, 0, 255).astype(np.uint8)

# ==========================================
# SAVE RESULTS
# ==========================================

result_path = OUT / "v37_edge_cleanup_result.png"
mask_output = OUT / "v37_cleanup_mask.png"

if not cv2.imwrite(str(result_path), final):
    raise RuntimeError("Failed to save V3.7 result")

cv2.imwrite(str(mask_output), clean_mask)

print("VIRTUALFIT V3.7 COMPLETE!")
print("Saved:", result_path)
print("Mask:", mask_output)

cv2.imshow("VIRTUALFIT V3.7", final)

print("Press Q to close.")

while True:
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
