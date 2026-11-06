
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original = cv2.imread(str(OUT / f"capture_{SESSION}.png"))
ai = cv2.imread(str(OUT / "best_fitting_backup.png"))

if original is None or ai is None:
    raise FileNotFoundError("Original or AI image missing.")

h, w = original.shape[:2]
ai = cv2.resize(ai, (w, h))

mp_selfie = mp.solutions.selfie_segmentation

def segment_person(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_selfie.SelfieSegmentation(
        model_selection=1
    ) as segmenter:
        result = segmenter.process(rgb)

    if result.segmentation_mask is None:
        raise RuntimeError("Person segmentation failed.")

    mask = (
        result.segmentation_mask > 0.5
    ).astype(np.uint8) * 255

    return mask

print("Segmenting original photograph...")
original_mask = segment_person(original)

print("Segmenting AI photograph...")
ai_mask = segment_person(ai)

cv2.imwrite(
    str(OUT / "v45_original_person_mask.png"),
    original_mask
)

cv2.imwrite(
    str(OUT / "v45_ai_person_mask.png"),
    ai_mask
)

def overlay(image, mask):
    result = image.copy()
    result[mask > 0] = (
        result[mask > 0] * 0.6
        + np.array([0, 255, 0]) * 0.4
    ).astype(np.uint8)
    return result

comparison = np.hstack([
    overlay(original, original_mask),
    overlay(ai, ai_mask)
])

output = OUT / "v45_person_comparison.png"

cv2.imwrite(str(output), comparison)

print("\nVIRTUALFIT V4.5 COMPLETE!")
print("Original person pixels:", cv2.countNonZero(original_mask))
print("AI person pixels:", cv2.countNonZero(ai_mask))
print("Saved:", output)
print("No changes made to the blue T-shirt.")
