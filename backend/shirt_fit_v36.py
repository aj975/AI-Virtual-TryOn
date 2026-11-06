
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import pipeline

# ==========================================
# VIRTUALFIT V3.6 - SMOOTH BODY FITTING
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "outputs"

SESSION = "8658aeb79b8c4897903ad0094bf8a262"

original_path = OUTPUT / f"capture_{SESSION}.png"
ai_path = OUTPUT / f"web_tryon_{SESSION}.png"
mask_path = OUTPUT / "v3_final_mask_v33.png"

original = cv2.imread(str(original_path))
ai = cv2.imread(str(ai_path))
shirt_mask = cv2.imread(
    str(mask_path),
    cv2.IMREAD_GRAYSCALE
)

if any(x is None for x in [original, ai, shirt_mask]):
    raise FileNotFoundError(
        "Required images or mask missing."
    )

h, w = original.shape[:2]

ai = cv2.resize(ai, (w, h))

shirt_mask = cv2.resize(
    shirt_mask,
    (w, h),
    interpolation=cv2.INTER_NEAREST
)

# ==========================================
# DETECT AI SHIRT
# ==========================================

print("Loading shirt segmentation model...", flush=True)

segmenter = pipeline(
    "image-segmentation",
    model="mattmdjaga/segformer_b0_clothes",
    device=-1
)

ai_pil = Image.fromarray(
    cv2.cvtColor(ai, cv2.COLOR_BGR2RGB)
)

results = segmenter(ai_pil)

ai_mask = np.zeros((h, w), dtype=np.uint8)

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

        ai_mask = np.maximum(ai_mask, mask)

if not np.any(ai_mask):
    raise RuntimeError(
        "AI shirt not detected."
    )

# ==========================================
# MEASURE SHIRT WIDTHS
# ==========================================

original_binary = shirt_mask > 127
ai_binary = ai_mask > 127

ys = np.where(original_binary)[0]

if len(ys) == 0:
    raise RuntimeError("Original shirt mask is empty.")

top = int(ys.min())
bottom = int(ys.max())

row_scales = np.ones(h, dtype=np.float32)
row_centers = np.full(
    h,
    w / 2,
    dtype=np.float32
)

valid_rows = []

for y in range(top, bottom + 1):

    original_x = np.where(
        original_binary[y]
    )[0]

    ai_x = np.where(
        ai_binary[y]
    )[0]

    if len(original_x) < 20 or len(ai_x) < 20:
        continue

    ow = original_x[-1] - original_x[0]
    aw = ai_x[-1] - ai_x[0]

    if aw <= 0:
        continue

    ratio = np.clip(
        ow / aw,
        0.90,
        1.03
    )

    progress = (
        (y - top) / max(bottom - top, 1)
    )

    # Gentle fitting at shoulders,
    # stronger fitting near the waist
    strength = 0.05 + 0.30 * progress

    row_scales[y] = (
        1.0 + (ratio - 1.0) * strength
    )

    row_centers[y] = (
        original_x[0] + original_x[-1]
    ) / 2

    valid_rows.append(y)

if len(valid_rows) < 2:
    raise RuntimeError(
        "Not enough shirt rows detected."
    )

# ==========================================
# SMOOTH THE FITTING CURVE
# ==========================================

valid_rows = np.array(valid_rows)
rows = np.arange(h)

row_scales = np.interp(
    rows,
    valid_rows,
    row_scales[valid_rows]
).astype(np.float32)

row_centers = np.interp(
    rows,
    valid_rows,
    row_centers[valid_rows]
).astype(np.float32)

# Smooth vertical changes to avoid zigzags
row_scales = cv2.GaussianBlur(
    row_scales.reshape(-1, 1),
    (1, 81),
    0
).reshape(-1)

row_centers = cv2.GaussianBlur(
    row_centers.reshape(-1, 1),
    (1, 81),
    0
).reshape(-1)

# ==========================================
# CREATE SMOOTH IMAGE WARP
# ==========================================

grid_x = np.tile(
    np.arange(w, dtype=np.float32),
    (h, 1)
)

grid_y = np.tile(
    np.arange(h, dtype=np.float32)[:, None],
    (1, w)
)

scale_map = row_scales[:, None]
center_map = row_centers[:, None]

map_x = (
    (grid_x - center_map) / scale_map
    + center_map
).astype(np.float32)

warped = cv2.remap(
    ai,
    map_x,
    grid_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_REPLICATE
)

# ==========================================
# BLEND INTO ORIGINAL PHOTO
# ==========================================

smooth_mask = cv2.GaussianBlur(
    shirt_mask,
    (9, 9),
    0
)

alpha = (
    smooth_mask.astype(np.float32) / 255.0
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

output_path = (
    OUTPUT / "v36_smooth_fitted_result.png"
)

saved = cv2.imwrite(
    str(output_path),
    final
)

if not saved:
    raise RuntimeError("Could not save result image.")

print("\nVIRTUALFIT V3.6 COMPLETE!", flush=True)
print(f"Saved: {output_path}", flush=True)

# ==========================================
# DISPLAY RESULT
# ==========================================

cv2.imshow(
    "VIRTUALFIT V3.6 - Smooth Fit",
    final
)

print("Press Q to close.")

while True:
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
