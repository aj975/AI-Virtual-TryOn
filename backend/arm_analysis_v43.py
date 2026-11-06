
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# VIRTUALFIT V4.3 - ARM LANDMARK ANALYSIS

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"

SESSION = "b4e2dcb16d774017b70d9497b39d7cc7"

original = cv2.imread(str(OUT / f"capture_{SESSION}.png"))
ai = cv2.imread(str(OUT / "best_fitting_backup.png"))

if original is None or ai is None:
    raise FileNotFoundError("Original or AI image missing.")

h, w = original.shape[:2]
ai = cv2.resize(ai, (w, h))

mp_pose = mp.solutions.pose

landmark_ids = {
    "Left shoulder": 11,
    "Right shoulder": 12,
    "Left elbow": 13,
    "Right elbow": 14,
    "Left wrist": 15,
    "Right wrist": 16
}

def analyze(image, title):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    marked = image.copy()

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose:
        result = pose.process(rgb)

    if not result.pose_landmarks:
        print(f"{title}: Pose not detected.")
        return marked, None

    points = {}

    for name, idx in landmark_ids.items():
        lm = result.pose_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)

        points[name] = (x, y, lm.visibility)

        if lm.visibility > 0.5:
            cv2.circle(marked, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                marked, name,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 255, 255), 1
            )

    for side in ["Left", "Right"]:
        shoulder = points[f"{side} shoulder"]
        elbow = points[f"{side} elbow"]
        wrist = points[f"{side} wrist"]

        if min(shoulder[2], elbow[2], wrist[2]) > 0.5:
            cv2.line(
                marked, shoulder[:2], elbow[:2],
                (0, 255, 255), 2
            )
            cv2.line(
                marked, elbow[:2], wrist[:2],
                (0, 255, 255), 2
            )

    print(f"\n{title}")
    for name, (x, y, visibility) in points.items():
        print(f"{name}: ({x}, {y}), confidence={visibility:.2f}")

    return marked, points

original_marked, original_points = analyze(original, "ORIGINAL")
ai_marked, ai_points = analyze(ai, "AI GENERATED")

comparison = np.hstack([original_marked, ai_marked])

output = OUT / "v43_arm_landmarks.png"
cv2.imwrite(str(output), comparison)

print("\nSaved:", output)
print("Blue T-shirt has not been modified.")
