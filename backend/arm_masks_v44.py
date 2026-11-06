
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import pipeline

# VIRTUALFIT V4.4 - ARM SEGMENTATION

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original = cv2.imread(str(OUT / f"capture_{SESSION}.png"))
ai = cv2.imread(str(OUT / "best_fitting_backup.png"))

if original is None or ai is None:
    raise FileNotFoundError("Original or AI image missing.")

h, w = original.shape[:2]
ai = cv2.resize(ai, (w, h))

print("Loading clothing segmentation model...", flush=True)

segmenter = pipeline(
    "image-segmentation",
    model="mattmdjaga/segformer_b0_clothes",
    device=-1
)

def get_arm_mask(image, title):
    pil = Image.fromarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )

    results = segmenter(pil)

    mask = np.zeros((h, w), dtype=np.uint8)

    for item in results:
        if item["label"] in ["Left-arm", "Right-arm"]:
            part = np.array(item["mask"].convert("L"))

            if part.shape != (h, w):
                part = cv2.resize(
                    part, (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

            mask = np.maximum(mask, part)

    mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    print(
        f"{title} arm pixels:",
        cv2.countNonZero(mask)
    )

    return mask

print("Detecting original arms...", flush=True)
original_mask = get_arm_mask(original, "Original")

print("Detecting AI arms...", flush=True)
ai_mask = get_arm_mask(ai, "AI")

# Save binary masks
cv2.imwrite(str(OUT / "v44_original_arms.png"), original_mask)
cv2.imwrite(str(OUT / "v44_ai_arms.png"), ai_mask)

# Draw arm masks as green overlays
def visualize(image, mask):
    overlay = image.copy()
    overlay[mask > 0] = (0, 255, 0)

    return cv2.addWeighted(
        image, 0.65,
        overlay, 0.35,
        0
    )

comparison = np.hstack([
    visualize(original, original_mask),
    visualize(ai, ai_mask)
])

output = OUT / "v44_arm_comparison.png"

cv2.imwrite(str(output), comparison)

print("\nVIRTUALFIT V4.4 COMPLETE!")
print("Saved:", output)
print("No changes made to the blue T-shirt.")
