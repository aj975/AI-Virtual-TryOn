
import cv2
import numpy as np
from pathlib import Path
from transformers import pipeline

# ==========================================
# VIRTUALFIT V3 - CLOTHING MASK DETECTION
# ==========================================

BASE = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    BASE / "outputs" /
    "capture_8658aeb79b8c4897903ad0094bf8a262.png"
)

OUTPUT = BASE / "outputs"
OUTPUT.mkdir(exist_ok=True)

# ==========================================
# LOAD ORIGINAL PHOTO
# ==========================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not find webcam photo: {IMAGE_PATH}"
    )

height, width = image.shape[:2]

print("Original photo loaded successfully!", flush=True)

# ==========================================
# LOAD CLOTHING SEGMENTATION AI
# ==========================================

print("Loading clothing segmentation model...", flush=True)

model = pipeline(
    "image-segmentation",
    model="mattmdjaga/segformer_b0_clothes",
    device=-1
)

print("Detecting shirt and arms...", flush=True)

results = model(str(IMAGE_PATH))

# ==========================================
# CREATE MASKS
# ==========================================

shirt_mask = np.zeros(
    (height, width), dtype=np.uint8
)

arms_mask = np.zeros(
    (height, width), dtype=np.uint8
)

for item in results:

    label = item["label"]

    mask = np.array(
        item["mask"].convert("L")
    )

    mask = cv2.resize(
        mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    if label == "Upper-clothes":
        shirt_mask = np.maximum(shirt_mask, mask)

    elif label in ["Left-arm", "Right-arm"]:
        arms_mask = np.maximum(arms_mask, mask)

# ==========================================
# SAVE MASKS
# ==========================================

combined_mask = np.maximum(
    shirt_mask,
    arms_mask
)

cv2.imwrite(
    str(OUTPUT / "v3_shirt_mask.png"),
    shirt_mask
)

cv2.imwrite(
    str(OUTPUT / "v3_arms_mask.png"),
    arms_mask
)

cv2.imwrite(
    str(OUTPUT / "v3_combined_mask.png"),
    combined_mask
)

# ==========================================
# CREATE VISUAL PREVIEW
# ==========================================

preview = image.copy()

overlay = np.zeros_like(image)
overlay[:, :] = (0, 255, 0)

alpha = (shirt_mask.astype(np.float32) / 255.0)[:, :, None] * 0.4

preview = (
    preview.astype(np.float32) * (1 - alpha)
    + overlay.astype(np.float32) * alpha
).astype(np.uint8)

cv2.imwrite(
    str(OUTPUT / "v3_shirt_preview.png"),
    preview
)

print("V3 MASKS SAVED SUCCESSFULLY!", flush=True)
print("Green area = detected T-shirt", flush=True)
print("Press Q to close the windows.", flush=True)

# ==========================================
# DISPLAY RESULTS
# ==========================================

cv2.imshow("Original Webcam Photo", image)
cv2.imshow("V3 Shirt Mask", shirt_mask)
cv2.imshow("V3 Arms Mask", arms_mask)
cv2.imshow("V3 Shirt Preview", preview)

while True:
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
