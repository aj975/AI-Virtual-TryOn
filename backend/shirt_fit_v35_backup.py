
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import pipeline

# ==========================================
# VIRTUALFIT V3.5 - NATURAL SHIRT FITTING
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "outputs"

SESSION = "8658aeb79b8c4897903ad0094bf8a262"

original_path = OUTPUT / f"capture_{SESSION}.png"
ai_path = OUTPUT / f"web_tryon_{SESSION}.png"
mask_path = OUTPUT / "v3_final_mask_v33.png"

# ==========================================
# LOAD IMAGES
# ==========================================

original = cv2.imread(str(original_path))
ai = cv2.imread(str(ai_path))
original_mask = cv2.imread(
    str(mask_path),
    cv2.IMREAD_GRAYSCALE
)

if original is None:
    raise FileNotFoundError(
        f"Original photo not found: {original_path}"
    )

if ai is None:
    raise FileNotFoundError(
        f"AI photo not found: {ai_path}"
    )

if original_mask is None:
    raise FileNotFoundError(
        f"Shirt mask not found: {mask_path}"
    )

h, w = original.shape[:2]

ai = cv2.resize(
    ai,
    (w, h),
    interpolation=cv2.INTER_LINEAR
)

original_mask = cv2.resize(
    original_mask,
    (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# DETECT AI SHIRT SILHOUETTE
# ==========================================

print("Loading clothing segmentation model...", flush=True)

segmenter = pipeline(
    "image-segmentation",
    model="mattmdjaga/segformer_b0_clothes",
    device=-1
)

print("Detecting AI shirt...", flush=True)

# FIX: Convert OpenCV image to PIL image
ai_rgb = cv2.cvtColor(
    ai,
    cv2.COLOR_BGR2RGB
)

ai_pil = Image.fromarray(ai_rgb)

results = segmenter(ai_pil)

ai_mask = np.zeros(
    (h, w),
    dtype=np.uint8
)

for item in results:

    if item["label"] == "Upper-clothes":

        mask = np.array(
            item["mask"].convert("L")
        )

        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        ai_mask = np.maximum(
            ai_mask,
            mask
        )

if not np.any(ai_mask):
    raise RuntimeError(
        "AI shirt could not be detected."
    )

print("AI shirt detected successfully!", flush=True)

# ==========================================
# CALCULATE SHIRT WIDTH BY ROW
# ==========================================

original_binary = original_mask > 127
ai_binary = ai_mask > 127

ys, xs = np.where(original_binary)

if len(ys) == 0:
    raise RuntimeError(
        "Original shirt mask is empty."
    )

top = int(ys.min())
bottom = int(ys.max())

# ==========================================
# GENTLE ROW-BY-ROW FIT ADJUSTMENT
# ==========================================

warped = ai.copy()

for y in range(top, bottom + 1):

    original_x = np.where(
        original_binary[y]
    )[0]

    ai_x = np.where(
        ai_binary[y]
    )[0]

    if len(original_x) < 10 or len(ai_x) < 10:
        continue

    original_left = original_x.min()
    original_right = original_x.max()

    ai_left = ai_x.min()
    ai_right = ai_x.max()

    original_width = (
        original_right - original_left
    )

    ai_width = (
        ai_right - ai_left
    )

    if ai_width <= 0:
        continue

    # Limit changes to avoid strong distortion
    target_scale = np.clip(
        original_width / ai_width,
        0.88,
        1.05
    )

    # Gradually increase adjustment toward waist
    progress = (
        (y - top) / max(bottom - top, 1)
    )

    strength = 0.15 + 0.45 * progress

    scale = (
        1.0 + (target_scale - 1.0) * strength
    )

    source_center = (
        ai_left + ai_right
    ) / 2.0

    target_center = (
        original_left + original_right
    ) / 2.0

    x_coords = np.arange(
        w,
        dtype=np.float32
    )

    map_x = (
        (x_coords - target_center) / scale
        + source_center
    ).astype(np.float32)

    map_y = np.full(
        w,
        y,
        dtype=np.float32
    )

    warped[y:y+1] = cv2.remap(
        ai,
        map_x.reshape(1, -1),
        map_y.reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

# ==========================================
# BLEND WITH ORIGINAL PHOTO
# ==========================================

mask = cv2.GaussianBlur(
    original_mask,
    (9, 9),
    0
)

alpha = (
    mask.astype(np.float32) / 255.0
)[:, :, None]

final = (
    warped.astype(np.float32) * alpha
    + original.astype(np.float32) * (1 - alpha)
)

final = np.clip(
    final,
    0,
    255
).astype(np.uint8)

# ==========================================
# SAVE RESULTS
# ==========================================

ai_mask_output = OUTPUT / "v35_ai_shirt_mask.png"
result_output = OUTPUT / "v35_fitted_result.png"

cv2.imwrite(
    str(ai_mask_output),
    ai_mask
)

cv2.imwrite(
    str(result_output),
    final
)

print("VIRTUALFIT V3.5 COMPLETE!", flush=True)
print(f"Saved: {result_output}", flush=True)

# ==========================================
# DISPLAY RESULTS
# ==========================================

cv2.imshow(
    "AI Shirt Mask",
    ai_mask
)

cv2.imshow(
    "VIRTUALFIT V3.5 - Natural Fit",
    final
)

print("Press Q to close windows.", flush=True)

while True:
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

