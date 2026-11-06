
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# VIRTUALFIT V4.6 - ARM WIDTH ANALYSIS
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original = cv2.imread(str(OUT / f"capture_{SESSION}.png"))
ai = cv2.imread(str(OUT / "best_fitting_backup.png"))

original_mask = cv2.imread(
    str(OUT / "v45_original_person_mask.png"), 0
)
ai_mask = cv2.imread(
    str(OUT / "v45_ai_person_mask.png"), 0
)

if any(x is None for x in [
    original, ai, original_mask, ai_mask
]):
    raise FileNotFoundError("One or more input images are missing.")

h, w = original.shape[:2]

ai = cv2.resize(ai, (w, h))
original_mask = cv2.resize(
    original_mask, (w, h),
    interpolation=cv2.INTER_NEAREST
)
ai_mask = cv2.resize(
    ai_mask, (w, h),
    interpolation=cv2.INTER_NEAREST
)

# Approximate elbow locations from our V4.3 measurements.
# These help us locate the exposed arms.
points = {
    "original": {
        "image_left": (210, 389),
        "image_right": (505, 382)
    },
    "ai": {
        "image_left": (220, 372),
        "image_right": (509, 370)
    }
}

def measure_width(mask, x_center, y, search_radius=75):
    if y < 0 or y >= mask.shape[0]:
        return None

    row = mask[y] > 127

    left = max(0, x_center - search_radius)
    right = min(mask.shape[1], x_center + search_radius + 1)

    # Find the connected foreground run containing the
    # landmark or nearest foreground pixel.
    indices = np.flatnonzero(row[left:right])

    if len(indices) == 0:
        return None

    positions = indices + left
    nearest = int(positions[np.argmin(abs(positions - x_center))])

    start = nearest
    end = nearest

    while start > 0 and row[start - 1]:
        start -= 1

    while end < mask.shape[1] - 1 and row[end + 1]:
        end += 1

    return start, end, end - start + 1

def analyze(image, mask, elbow_points, label):
    marked = image.copy()
    measurements = {}

    for side, (x, elbow_y) in elbow_points.items():
        measurements[side] = []

        # Measure near elbow and farther down forearm.
        for offset in [-20, 0, 20, 40]:
            y = elbow_y + offset

            result = measure_width(mask, x, y)

            if result is None:
                continue

            left, right, width = result

            # Reject implausibly large runs, likely torso.
            if width > 100:
                continue

            cv2.line(
                marked,
                (left, y),
                (right, y),
                (0, 255, 255),
                2
            )

            cv2.putText(
                marked,
                f"{width}px",
                (left, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

            measurements[side].append((offset, width))

            print(
                f"{label} {side}, elbow offset "
                f"{offset:+d}: {width}px"
            )

    return marked, measurements

print("\nORIGINAL ARM MEASUREMENTS")
original_marked, original_data = analyze(
    original,
    original_mask,
    points["original"],
    "Original"
)

print("\nAI ARM MEASUREMENTS")
ai_marked, ai_data = analyze(
    ai,
    ai_mask,
    points["ai"],
    "AI"
)

comparison = np.hstack([
    original_marked,
    ai_marked
])

output = OUT / "v46_arm_width_comparison.png"

if not cv2.imwrite(str(output), comparison):
    raise RuntimeError("Could not save comparison.")

print("\nVIRTUALFIT V4.6 COMPLETE!")
print("Saved:", output)
print("No image pixels or T-shirt fitting were modified.")
