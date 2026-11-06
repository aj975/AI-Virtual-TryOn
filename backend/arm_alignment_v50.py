
import cv2
import numpy as np
from pathlib import Path

# VIRTUALFIT V5.0 - ARM ALIGNMENT DIAGNOSTIC
# Does not modify the blue T-shirt.

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original = cv2.imread(str(OUT / f"capture_{SESSION}.png"))
ai = cv2.imread(str(OUT / "best_fitting_backup.png"))

if original is None or ai is None:
    raise FileNotFoundError("Original or AI photograph missing.")

h, w = original.shape[:2]
ai = cv2.resize(ai, (w, h), interpolation=cv2.INTER_LINEAR)

# V4.3 elbow measurements
# Coordinates are already in the 640x480 reference frame.
original_elbows = {
    "image_left": (210, 389),
    "image_right": (505, 382)
}

ai_elbows = {
    "image_left": (220, 372),
    "image_right": (509, 370)
}

# Calculate alignment offsets.
offsets = {}

for side in original_elbows:
    ox, oy = original_elbows[side]
    ax, ay = ai_elbows[side]

    dx = ax - ox
    dy = ay - oy

    offsets[side] = (dx, dy)

    print(f"{side}: shift x={dx:+d}, y={dy:+d}")

# Visualize corresponding elbow landmarks.
preview = ai.copy()

for side in original_elbows:
    ox, oy = original_elbows[side]
    ax, ay = ai_elbows[side]

    # Green: original elbow
    cv2.circle(preview, (ox, oy), 7, (0, 255, 0), -1)

    # Red: AI elbow
    cv2.circle(preview, (ax, ay), 7, (0, 0, 255), -1)

    # Yellow: displacement
    cv2.arrowedLine(
        preview,
        (ox, oy),
        (ax, ay),
        (0, 255, 255),
        2,
        tipLength=0.25
    )

output = OUT / "v50_elbow_alignment.png"

if not cv2.imwrite(str(output), preview):
    raise RuntimeError("Could not save alignment preview.")

print("\nVIRTUALFIT V5.0 DIAGNOSTIC COMPLETE!")
print("Saved:", output)
print("No photographs were modified.")
