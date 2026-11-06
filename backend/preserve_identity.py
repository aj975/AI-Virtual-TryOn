
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import pipeline

# ==========================================
# VIRTUALFIT V4.1 - IDENTITY PRESERVATION
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "8658aeb79b8c4897903ad0094bf8a262"

original_path = OUT / f"capture_{SESSION}.png"
ai_path = OUT / f"web_tryon_{SESSION}.png"

original = cv2.imread(str(original_path))
ai = cv2.imread(str(ai_path))

if original is None or ai is None:
    raise FileNotFoundError(
        "Original or AI-generated photograph is missing."
    )

h, w = original.shape[:2]

ai = cv2.resize(
    ai, (w, h),
    interpolation=cv2.INTER_LINEAR
)

# ==========================================
# LOAD CLOTHING SEGMENTATION MODEL
# ==========================================

print("Loading clothing segmentation model...", flush=True)

segmenter = pipeline(
    "image-segmentation",
    model="mattmdjaga/segformer_b0_clothes",
    device=-1
)

# ==========================================
# SEGMENT ORIGINAL AND AI PHOTOS
# ==========================================

def get_clothing_mask(image):
    pil_image = Image.fromarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    results = segmenter(pil_image)

    mask = np.zeros(
        image.shape[:2], dtype=np.uint8
    )

    for item in results:
        if item["label"] == "Upper-clothes":
            part = np.array(
                item["mask"].convert("L")
            )

            part = cv2.resize(
                part, (w, h),
                interpolation=cv2.INTER_NEAREST
            )

            mask = np.maximum(mask, part)

    return mask

print("Detecting original shirt...", flush=True)
original_mask = get_clothing_mask(original)

print("Detecting AI-generated shirt...", flush=True)
ai_mask = get_clothing_mask(ai)

if not np.any(original_mask):
    raise RuntimeError("Original shirt not detected.")

if not np.any(ai_mask):
    raise RuntimeError("AI-generated shirt not detected.")

# ==========================================
# BUILD CONSERVATIVE CLOTHING MASK
# ==========================================

# Use original clothing region as our base.
# This avoids replacing the whole body/background.

mask = cv2.bitwise_and(
    original_mask, ai_mask
)

# Close small gaps within the shirt
mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (11, 11)
    )
)

# Feather edges for a smoother transition
mask = cv2.GaussianBlur(
    mask, (11, 11), 0
)

# ==========================================
# BLEND AI SHIRT WITH ORIGINAL PHOTO
# ==========================================

alpha = (
    mask.astype(np.float32) / 255.0
)[:, :, None]

final = (
    ai.astype(np.float32) * alpha
    + original.astype(np.float32) * (1 - alpha)
)

final = np.clip(
    final, 0, 255
).astype(np.uint8)

# ==========================================
# SAVE OUTPUTS
# ==========================================

result_path = OUT / "v41_identity_preserved.png"
mask_path = OUT / "v41_clothing_mask.png"

if not cv2.imwrite(str(result_path), final):
    raise RuntimeError("Failed to save output image.")

cv2.imwrite(str(mask_path), mask)

print("\nVIRTUALFIT V4.1 COMPLETE!")
print("Saved:", result_path)
print("Mask:", mask_path)

# ==========================================
# DISPLAY
# ==========================================

cv2.imshow(
    "VIRTUALFIT V4.1 - Identity Preserved",
    final
)

print("Press any key in the image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
