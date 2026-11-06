import cv2
import numpy as np
from pathlib import Path
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "assets" / "person.jpeg"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading SegFormer-B0...", flush=True)

model = pipeline(
    "image-segmentation",
    model="mattmdjaga/segformer_b0_clothes",
    device=-1
)

print("Analysing your photo...", flush=True)

results = model(str(IMAGE_PATH))

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

overlay = image.copy()

colors = {
    "Background": (80, 80, 80),
    "Hair": (255, 0, 255),
    "Upper-clothes": (0, 255, 0),
    "Pants": (255, 100, 0),
    "Dress": (0, 200, 255),
    "Belt": (255, 255, 0),
    "Face": (0, 180, 255),
    "Left-leg": (180, 0, 255),
    "Right-leg": (255, 0, 180),
    "Left-arm": (0, 0, 255),
    "Right-arm": (255, 255, 0),
    "Bag": (100, 255, 100),
}

for item in results:
    label = item["label"]

    if label == "Background":
        continue

    mask = np.array(item["mask"].convert("L"))
    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    region = mask > 127
    color = colors.get(label, (255, 255, 255))

    overlay[region] = (
        0.45 * overlay[region] + 0.55 * np.array(color)
    ).astype(np.uint8)

    ys, xs = np.where(region)

    if len(xs) > 0:
        x = int(np.mean(xs))
        y = int(np.mean(ys))
        cv2.putText(
            overlay, label, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

output_path = OUTPUT_DIR / "segformer_visualization.png"
cv2.imwrite(str(output_path), overlay)

print("VISUALIZATION SAVED!")
print(output_path)

cv2.imshow("Original Photo", image)
cv2.imshow("AI Clothing Recognition", overlay)

print("Press Q to close the windows.")

while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
