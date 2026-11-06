
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V3.3 - TARGETED SHIRT CORRECTION
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "outputs"

SESSION = "8658aeb79b8c4897903ad0094bf8a262"

original_path = OUTPUT / f"capture_{SESSION}.png"
ai_path = OUTPUT / f"web_tryon_{SESSION}.png"
mask_path = OUTPUT / "v3_filled_shirt_mask_v32.png"
arms_path = OUTPUT / "v3_arms_mask.png"

original = cv2.imread(str(original_path))
ai_image = cv2.imread(str(ai_path))
shirt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
arms_mask = cv2.imread(str(arms_path), cv2.IMREAD_GRAYSCALE)

if any(x is None for x in [original, ai_image, shirt_mask, arms_mask]):
    raise FileNotFoundError(
        "One or more required V3 images or masks are missing."
    )

height, width = original.shape[:2]

ai_image = cv2.resize(ai_image, (width, height))
shirt_mask = cv2.resize(
    shirt_mask, (width, height),
    interpolation=cv2.INTER_NEAREST
)
arms_mask = cv2.resize(
    arms_mask, (width, height),
    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# DETECT REMAINING YELLOW FABRIC
# ==========================================

hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

# Detect yellow shirt pixels
yellow_mask = cv2.inRange(
    hsv,
    np.array([15, 45, 45]),
    np.array([45, 255, 255])
)

# Only look near the existing shirt boundary
near_shirt = cv2.dilate(
    shirt_mask,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (35, 35)
    )
)

yellow_near_shirt = cv2.bitwise_and(
    yellow_mask, near_shirt
)

# Protect exposed arms
protected_arms = cv2.dilate(
    arms_mask,
    np.ones((9, 9), np.uint8)
)

yellow_near_shirt = cv2.bitwise_and(
    yellow_near_shirt,
    cv2.bitwise_not(protected_arms)
)

# ==========================================
# COMBINE MASKS
# ==========================================

final_mask = cv2.bitwise_or(
    shirt_mask,
    yellow_near_shirt
)

# Smooth small gaps
final_mask = cv2.morphologyEx(
    final_mask,
    cv2.MORPH_CLOSE,
    np.ones((7, 7), np.uint8)
)

# Feather edges
smooth_mask = cv2.GaussianBlur(
    final_mask, (9, 9), 0
)

# ==========================================
# BLEND IMAGES
# ==========================================

alpha = smooth_mask.astype(np.float32) / 255.0
alpha = alpha[:, :, None]

result = (
    ai_image.astype(np.float32) * alpha
    + original.astype(np.float32) * (1 - alpha)
)

result = np.clip(result, 0, 255).astype(np.uint8)

# ==========================================
# SAVE RESULTS
# ==========================================

result_path = OUTPUT / "v3_blended_result_v33.png"
mask_output = OUTPUT / "v3_final_mask_v33.png"

cv2.imwrite(str(result_path), result)
cv2.imwrite(str(mask_output), final_mask)

print("VIRTUALFIT V3.3 COMPLETE!")
print(f"Result saved: {result_path}")

# ==========================================
# DISPLAY RESULT
# ==========================================

cv2.imshow("V3.3 Final Mask", final_mask)
cv2.imshow("VIRTUALFIT V3.3 - Result", result)

print("Press Q to close windows.")

while True:
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
