
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V3.2 - IMPROVED CLOTHING BLENDING
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "outputs"

SESSION = "8658aeb79b8c4897903ad0094bf8a262"

original_path = OUTPUT / f"capture_{SESSION}.png"
ai_path = OUTPUT / f"web_tryon_{SESSION}.png"
mask_path = OUTPUT / "v3_shirt_mask.png"

# ==========================================
# LOAD IMAGES
# ==========================================

original = cv2.imread(str(original_path))
ai_image = cv2.imread(str(ai_path))
shirt_mask = cv2.imread(
    str(mask_path),
    cv2.IMREAD_GRAYSCALE
)

if original is None:
    raise FileNotFoundError(
        f"Original photo not found: {original_path}"
    )

if ai_image is None:
    raise FileNotFoundError(
        f"AI-generated photo not found: {ai_path}"
    )

if shirt_mask is None:
    raise FileNotFoundError(
        f"Shirt mask not found: {mask_path}"
    )

height, width = original.shape[:2]

print("All images loaded successfully!", flush=True)

# ==========================================
# MATCH IMAGE DIMENSIONS
# ==========================================

ai_image = cv2.resize(
    ai_image,
    (width, height),
    interpolation=cv2.INTER_LINEAR
)

shirt_mask = cv2.resize(
    shirt_mask,
    (width, height),
    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# CLEAN AND REPAIR SHIRT MASK
# ==========================================

_, shirt_mask = cv2.threshold(
    shirt_mask,
    127,
    255,
    cv2.THRESH_BINARY
)

# Close larger gaps in the shirt mask
large_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (35, 35)
)

shirt_mask = cv2.morphologyEx(
    shirt_mask,
    cv2.MORPH_CLOSE,
    large_kernel
)

# ==========================================
# FIND AND FILL SHIRT REGION
# ==========================================

contours, _ = cv2.findContours(
    shirt_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    raise RuntimeError(
        "No shirt detected in the mask."
    )

filled_mask = np.zeros_like(shirt_mask)

largest_contour = max(
    contours,
    key=cv2.contourArea
)

cv2.drawContours(
    filled_mask,
    [largest_contour],
    -1,
    255,
    thickness=cv2.FILLED
)

# Smooth small imperfections
small_kernel = np.ones(
    (5, 5),
    dtype=np.uint8
)

filled_mask = cv2.morphologyEx(
    filled_mask,
    cv2.MORPH_CLOSE,
    small_kernel
)

# ==========================================
# FEATHER SHIRT EDGES
# ==========================================

smooth_mask = cv2.GaussianBlur(
    filled_mask,
    (11, 11),
    0
)

# ==========================================
# BLEND AI SHIRT WITH ORIGINAL PHOTO
# ==========================================

alpha = (
    smooth_mask.astype(np.float32) / 255.0
)[:, :, None]

result = (
    ai_image.astype(np.float32) * alpha
    + original.astype(np.float32) * (1.0 - alpha)
)

result = np.clip(
    result,
    0,
    255
).astype(np.uint8)

# ==========================================
# SAVE RESULTS
# ==========================================

mask_output = OUTPUT / "v3_filled_shirt_mask_v32.png"
result_output = OUTPUT / "v3_blended_result_v32.png"

cv2.imwrite(
    str(mask_output),
    filled_mask
)

cv2.imwrite(
    str(result_output),
    result
)

print("VIRTUALFIT V3.2 BLENDING COMPLETE!", flush=True)
print(f"Filled mask: {mask_output}", flush=True)
print(f"Final result: {result_output}", flush=True)

# ==========================================
# DISPLAY RESULTS
# ==========================================

cv2.imshow(
    "Original Photo",
    original
)

cv2.imshow(
    "AI Generated Photo",
    ai_image
)

cv2.imshow(
    "V3.2 Filled Shirt Mask",
    filled_mask
)

cv2.imshow(
    "VIRTUALFIT V3.2 - Blended Result",
    result
)

print("Press Q to close all windows.", flush=True)

while True:
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
