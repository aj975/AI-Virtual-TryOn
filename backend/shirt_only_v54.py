
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import pipeline

# ==========================================
# VIRTUALFIT V5.4 - SHIRT ONLY COMPOSITING
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original = cv2.imread(
    str(OUT / f"capture_{SESSION}.png")
)

ai = cv2.imread(
    str(OUT / "best_fitting_backup.png")
)

if original is None or ai is None:
    raise FileNotFoundError(
        "Original photo or best-fitting AI image missing."
    )

h, w = original.shape[:2]

ai = cv2.resize(
    ai, (w, h),
    interpolation=cv2.INTER_LINEAR
)

print("Loading clothing segmentation model...")

segmenter = pipeline(
    "image-segmentation",
    model="mattmdjaga/segformer_b0_clothes",
    device=-1
)

def get_shirt_mask(image):
    pil_image = Image.fromarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    results = segmenter(pil_image)

    mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    for item in results:
        label = item["label"].lower()

        if "upper-clothes" in label or "upper clothes" in label:
            part = np.array(
                item["mask"].convert("L")
            )

            if part.shape != mask.shape:
                part = cv2.resize(
                    part,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            mask = cv2.bitwise_or(
                mask, part
            )

    return mask

print("Detecting original shirt...")
original_mask = get_shirt_mask(original)

print("Detecting blue AI shirt...")
ai_mask = get_shirt_mask(ai)

# ==========================================
# COMBINE SHIRT MASKS
# ==========================================

combined = cv2.bitwise_or(
    original_mask,
    ai_mask
)

# Fill small holes without excessively
# expanding the shirt boundary.
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9)
)

combined = cv2.morphologyEx(
    combined,
    cv2.MORPH_CLOSE,
    kernel
)

# Remove small isolated regions.
count, labels, stats, _ = (
    cv2.connectedComponentsWithStats(
        (combined > 127).astype(np.uint8),
        connectivity=8
    )
)

clean = np.zeros_like(combined)

if count > 1:
    largest = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    clean[labels == largest] = 255

# ==========================================
# FEATHER SHIRT BOUNDARY
# ==========================================

alpha = cv2.GaussianBlur(
    clean.astype(np.float32) / 255.0,
    (9, 9),
    0
)

alpha = alpha[:, :, None]

result = (
    ai.astype(np.float32) * alpha
    + original.astype(np.float32) * (1 - alpha)
)

result = np.clip(
    result, 0, 255
).astype(np.uint8)

# ==========================================
# SAVE RESULTS
# ==========================================

mask_path = OUT / "v54_shirt_mask.png"
result_path = OUT / "v54_shirt_only_result.png"
comparison_path = OUT / "v54_before_after.png"

cv2.imwrite(str(mask_path), clean)
cv2.imwrite(str(result_path), result)

comparison = np.hstack([
    original,
    result
])

cv2.imwrite(
    str(comparison_path),
    comparison
)

print("\nVIRTUALFIT V5.4 COMPLETE!")
print("Saved:", result_path)
print("Saved:", comparison_path)
print("Original and approved AI images unchanged.")

