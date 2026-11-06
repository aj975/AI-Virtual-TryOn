import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import pipeline

BASE = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE / "assets" / "person.jpeg"
OUTPUT = BASE / "outputs"
OUTPUT.mkdir(exist_ok=True)

print("Loading clothing segmentation model...", flush=True)

model = pipeline(
    "image-segmentation",
    model="mattmdjaga/segformer_b0_clothes",
    device=-1
)

print("Analysing image...", flush=True)

results = model(str(IMAGE_PATH))

image = cv2.imread(str(IMAGE_PATH))
height, width = image.shape[:2]

shirt_mask = np.zeros((height, width), dtype=np.uint8)
arms_mask = np.zeros((height, width), dtype=np.uint8)

for item in results:
    label = item["label"]
    mask = np.array(item["mask"].convert("L"))
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    if label == "Upper-clothes":
        shirt_mask = np.maximum(shirt_mask, mask)

    if label in ["Left-arm", "Right-arm"]:
        arms_mask = np.maximum(arms_mask, mask)

combined_mask = np.maximum(shirt_mask, arms_mask)

cv2.imwrite(str(OUTPUT / "shirt_mask.png"), shirt_mask)
cv2.imwrite(str(OUTPUT / "arms_mask.png"), arms_mask)
cv2.imwrite(str(OUTPUT / "combined_body_mask.png"), combined_mask)

print("MASKS SAVED SUCCESSFULLY!", flush=True)

cv2.imshow("Original Photo", image)
cv2.imshow("Shirt Mask", shirt_mask)
cv2.imshow("Arms Mask", arms_mask)
cv2.imshow("Combined Mask", combined_mask)

print("Press Q to close.")

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
