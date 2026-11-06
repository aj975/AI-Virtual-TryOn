import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "outputs"

image = cv2.imread(str(BASE / "assets" / "person.jpeg"))
shirt_mask = cv2.imread(str(OUTPUT / "shirt_mask.png"), 0)
arms_mask = cv2.imread(str(OUTPUT / "arms_mask.png"), 0)

if image is None or shirt_mask is None or arms_mask is None:
    raise FileNotFoundError("Missing input image or masks.")

# Smooth small holes and irregular edges in the shirt mask
kernel = np.ones((5, 5), np.uint8)
shirt_mask = cv2.morphologyEx(
    shirt_mask, cv2.MORPH_CLOSE, kernel
)

# Protect exposed arms
shirt_mask[arms_mask > 127] = 0

# Convert to HSV to preserve brightness and fabric shadows
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Blue hue in OpenCV HSV
hsv[:, :, 0][shirt_mask > 127] = 110
hsv[:, :, 1][shirt_mask > 127] = 175

recolored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Blend edges for a smoother appearance
alpha = cv2.GaussianBlur(shirt_mask, (7, 7), 0)
alpha = alpha.astype(np.float32) / 255.0
alpha = alpha[:, :, None]

result = (
    image.astype(np.float32) * (1 - alpha)
    + recolored.astype(np.float32) * alpha
).astype(np.uint8)

output_path = OUTPUT / "blue_shirt_recolor.png"
cv2.imwrite(str(output_path), result)

print("SUCCESS! Shirt recolouring completed.")
print(output_path)
