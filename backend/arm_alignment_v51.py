
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V5.1 - ARM ALIGNMENT DIAGNOSTIC
# No changes to original or AI photographs
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original = cv2.imread(str(OUT / f"capture_{SESSION}.png"))
ai = cv2.imread(str(OUT / "best_fitting_backup.png"))

if original is None or ai is None:
    raise FileNotFoundError("Original or AI photograph missing.")

h, w = original.shape[:2]
ai = cv2.resize(ai, (w, h), interpolation=cv2.INTER_LINEAR)

# Landmark positions from our V4.3 MediaPipe analysis.
# Left/right refer to positions in the image.

arms = {
    "image_left": {
        "original_shoulder": (253, 237),
        "original_elbow": (210, 389),
        "ai_shoulder": (266, 228),
        "ai_elbow": (220, 372),
    },
    "image_right": {
        "original_shoulder": (466, 230),
        "original_elbow": (505, 382),
        "ai_shoulder": (463, 227),
        "ai_elbow": (509, 370),
    }
}

preview = ai.copy()

for side, points in arms.items():
    os = points["original_shoulder"]
    oe = points["original_elbow"]

    ais = points["ai_shoulder"]
    aie = points["ai_elbow"]

    # Green: original shoulder-to-elbow line
    cv2.line(preview, os, oe, (0, 255, 0), 3)
    cv2.circle(preview, os, 6, (0, 255, 0), -1)
    cv2.circle(preview, oe, 6, (0, 255, 0), -1)

    # Red: AI shoulder-to-elbow line
    cv2.line(preview, ais, aie, (0, 0, 255), 2)
    cv2.circle(preview, ais, 5, (0, 0, 255), -1)
    cv2.circle(preview, aie, 5, (0, 0, 255), -1)

    # Yellow arrows show landmark displacement
    cv2.arrowedLine(
        preview, os, ais,
        (0, 255, 255), 2, tipLength=0.25
    )
    cv2.arrowedLine(
        preview, oe, aie,
        (0, 255, 255), 2, tipLength=0.25
    )

    shoulder_shift = (
        ais[0] - os[0],
        ais[1] - os[1]
    )

    elbow_shift = (
        aie[0] - oe[0],
        aie[1] - oe[1]
    )

    print(f"\n{side.upper()}")
    print("Shoulder shift:", shoulder_shift)
    print("Elbow shift:", elbow_shift)

# ==========================================
# SAVE DIAGNOSTIC
# ==========================================

output = OUT / "v51_upper_arm_alignment.png"

if not cv2.imwrite(str(output), preview):
    raise RuntimeError("Could not save the diagnostic image.")

print("\nVIRTUALFIT V5.1 COMPLETE!")
print("Saved:", output)
print("Blue T-shirt and existing photographs unchanged.")
