import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

person = cv2.imread(str(BASE / "assets/person.jpeg"))
garment = cv2.imread(
    str(BASE / "garments/blue_tshirt_transparent.png"),
    cv2.IMREAD_UNCHANGED
)
arms = cv2.imread(str(OUT / "arms_mask.png"), 0)

if person is None or garment is None:
    raise FileNotFoundError("Person or garment image missing.")

if garment.ndim != 3 or garment.shape[2] != 4:
    raise ValueError("Garment must have an alpha channel.")

# Crop transparent margins
alpha = garment[:, :, 3]
ys, xs = np.where(alpha > 20)

if len(xs) == 0:
    raise ValueError("Garment image is fully transparent.")

garment = garment[
    ys.min():ys.max()+1,
    xs.min():xs.max()+1
]

# Detect body pose
pose = mp.solutions.pose.Pose(static_image_mode=True)
rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
result = pose.process(rgb)
pose.close()

if not result.pose_landmarks:
    raise RuntimeError("Pose not detected.")

h, w = person.shape[:2]
lm = result.pose_landmarks.landmark

def point(i):
    return np.array([lm[i].x*w, lm[i].y*h])

left_shoulder = point(11)
right_shoulder = point(12)
left_hip = point(23)
right_hip = point(24)

shoulder_center = (left_shoulder + right_shoulder) / 2
hip_center = (left_hip + right_hip) / 2

shoulder_width = np.linalg.norm(
    left_shoulder - right_shoulder
)
torso_height = np.linalg.norm(
    hip_center - shoulder_center
)

# Approximate garment placement
target_width = int(shoulder_width * 1.45)
target_height = int(torso_height * 1.35)

target_width = max(1, target_width)
target_height = max(1, target_height)

shirt = cv2.resize(
    garment,
    (target_width, target_height),
    interpolation=cv2.INTER_LINEAR
)

# Place garment around shoulder centre
x = int(shoulder_center[0] - target_width/2)
y = int(shoulder_center[1] - target_height*0.12)

# Build a full-frame garment layer
layer = np.zeros((h, w, 4), dtype=np.uint8)

x1 = max(0, x)
y1 = max(0, y)
x2 = min(w, x + target_width)
y2 = min(h, y + target_height)

if x2 <= x1 or y2 <= y1:
    raise RuntimeError("Garment is outside the image.")

sx1 = x1 - x
sy1 = y1 - y
sx2 = sx1 + (x2-x1)
sy2 = sy1 + (y2-y1)

layer[y1:y2, x1:x2] = shirt[sy1:sy2, sx1:sx2]

# Preserve visible exposed arms
if arms is not None:
    arm_protection = cv2.dilate(
        arms,
        np.ones((5,5), np.uint8)
    )
    layer[:, :, 3][arm_protection > 127] = 0

# Alpha composite
a = layer[:, :, 3:4].astype(np.float32) / 255.0

fitted = (
    person.astype(np.float32)*(1-a)
    + layer[:, :, :3].astype(np.float32)*a
).astype(np.uint8)

path = OUT / "garment_fit_test.png"
cv2.imwrite(str(path), fitted)

print("GARMENT FIT TEST COMPLETE!")
print("Saved to:", path)
