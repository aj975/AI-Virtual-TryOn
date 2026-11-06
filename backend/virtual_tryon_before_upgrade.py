
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
SHIRT_PATH = BASE_DIR / "garments" / "blue_tshirt_transparent.png"

# ==========================================
# LOAD T-SHIRT IMAGE
# ==========================================

shirt_image = cv2.imread(
    str(SHIRT_PATH),
    cv2.IMREAD_UNCHANGED
)

if shirt_image is None:
    raise FileNotFoundError(
        f"Could not load T-shirt image: {SHIRT_PATH}"
    )

if shirt_image.shape[2] == 4:
    shirt_bgr = shirt_image[:, :, :3]
    alpha = shirt_image[:, :, 3]
else:
    raise ValueError(
        "The T-shirt image has no alpha channel. "
        "Please use blue_tshirt_transparent.png"
    )

# Remove transparent margins
ys, xs = np.where(alpha > 0)

if len(xs) == 0:
    raise ValueError("T-shirt image is fully transparent!")

x1, x2 = xs.min(), xs.max() + 1
y1, y2 = ys.min(), ys.max() + 1

shirt_bgr = shirt_bgr[y1:y2, x1:x2]
alpha = alpha[y1:y2, x1:x2]

print("Transparent blue T-shirt loaded successfully!")


# ==========================================
# APPLY SHIRT TO BODY
# ==========================================

def apply_shirt(frame, landmarks):

    h, w = frame.shape[:2]

    def point(index):
        p = landmarks[index]
        return np.array(
            [p.x * w, p.y * h],
            dtype=np.float32
        )

    # Body landmarks
    ls = point(11)
    rs = point(12)
    lh = point(23)
    rh = point(24)

    # Check visibility
    for i in [11, 12, 23, 24]:
        if landmarks[i].visibility < 0.5:
            return frame

    # Screen-left and screen-right
    shoulders = sorted([ls, rs], key=lambda p: p[0])
    hips = sorted([lh, rh], key=lambda p: p[0])

    sl, sr = shoulders
    hl, hr = hips

    shoulder_width = np.linalg.norm(sr - sl)

    if shoulder_width < 40:
        return frame

    # Neck position
    neck_center = (sl + sr) / 2

    # Torso dimensions
    torso_height = np.linalg.norm(
        ((hl + hr) / 2) - neck_center
    )

    if torso_height < 50:
        return frame

    # Garment dimensions
    shirt_width = shoulder_width * 1.35
    shirt_height = torso_height * 1.15

    # Destination points
    top_left = (
        neck_center[0] - shirt_width * 0.50,
        neck_center[1] - shoulder_width * 0.12
    )

    top_right = (
        neck_center[0] + shirt_width * 0.50,
        neck_center[1] - shoulder_width * 0.12
    )

    bottom_right = (
        hr[0] + shoulder_width * 0.08,
        neck_center[1] + shirt_height * 0.85
    )

    bottom_left = (
        hl[0] - shoulder_width * 0.08,
        neck_center[1] + shirt_height * 0.85
    )

    destination = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)

    sh, sw = alpha.shape

    source = np.array([
        [0, 0],
        [sw - 1, 0],
        [sw - 1, sh - 1],
        [0, sh - 1]
    ], dtype=np.float32)

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(
        source, destination
    )

    warped_shirt = cv2.warpPerspective(
        shirt_bgr, matrix, (w, h)
    )

    warped_alpha = cv2.warpPerspective(
        alpha, matrix, (w, h)
    )

    # Smooth transparent blending
    mask = warped_alpha.astype(
        np.float32
    ) / 255.0

    mask = mask[:, :, np.newaxis]

    result = (
        warped_shirt.astype(np.float32) * mask
        +
        frame.astype(np.float32) * (1 - mask)
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# ==========================================
# START MEDIAPIPE POSE DETECTION
# ==========================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    pose.close()
    raise RuntimeError("Could not open webcam!")

print("AI VIRTUAL TRY-ON STARTED!")
print("Press Q to quit.")

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # Convert frame for MediaPipe
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(rgb)

        # Apply T-shirt when body is detected
        if results.pose_landmarks:
            frame = apply_shirt(
                frame,
                results.pose_landmarks.landmark
            )

        # Display project information
        cv2.putText(
            frame,
            "AI VIRTUAL TRY-ON",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "OUTFIT: BLUE T-SHIRT",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "PRESS Q TO QUIT",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AI Virtual Try-On",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    print("Virtual Try-On closed.")
