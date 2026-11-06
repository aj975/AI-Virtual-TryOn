
import cv2
import mediapipe as mp
from pathlib import Path

# ==========================================
# VIRTUALFIT V3.4 - BODY MEASUREMENT
# ==========================================

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "outputs"

SESSION = "8658aeb79b8c4897903ad0094bf8a262"

IMAGE_PATH = OUTPUT / f"web_tryon_{SESSION}.png"

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Original photo not found: {IMAGE_PATH}"
    )

height, width = image.shape[:2]

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ==========================================
# DETECT BODY LANDMARKS
# ==========================================

print("Detecting body proportions...", flush=True)

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
) as pose:

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        raise RuntimeError(
            "Body landmarks not detected."
        )

    landmarks = results.pose_landmarks.landmark

    def point(index):
        p = landmarks[index]
        return (
            int(p.x * width),
            int(p.y * height)
        )

    left_shoulder = point(11)
    right_shoulder = point(12)
    left_hip = point(23)
    right_hip = point(24)

    # ======================================
    # CALCULATE APPROXIMATE PROPORTIONS
    # ======================================

    shoulder_width = abs(
        left_shoulder[0] - right_shoulder[0]
    )

    hip_width = abs(
        left_hip[0] - right_hip[0]
    )

    shoulder_center = (
        (left_shoulder[0] + right_shoulder[0]) // 2,
        (left_shoulder[1] + right_shoulder[1]) // 2
    )

    hip_center = (
        (left_hip[0] + right_hip[0]) // 2,
        (left_hip[1] + right_hip[1]) // 2
    )

    torso_length = abs(
        hip_center[1] - shoulder_center[1]
    )

    print("\nBODY MEASUREMENTS (PIXELS)")
    print("--------------------------")
    print(f"Shoulder width: {shoulder_width}")
    print(f"Hip width: {hip_width}")
    print(f"Torso length: {torso_length}")

    # ======================================
    # DRAW LANDMARKS
    # ======================================

    preview = image.copy()

    mp_draw.draw_landmarks(
        preview,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    for label, position in [
        ("LEFT SHOULDER", left_shoulder),
        ("RIGHT SHOULDER", right_shoulder),
        ("LEFT HIP", left_hip),
        ("RIGHT HIP", right_hip)
    ]:
        cv2.circle(
            preview,
            position,
            7,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            preview,
            label,
            (position[0] + 10, position[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

    # ======================================
    # SAVE PREVIEW
    # ======================================

    output_path = OUTPUT / "v34_body_landmarks.png"

    cv2.imwrite(str(output_path), preview)

    print(f"\nSaved preview: {output_path}")

    cv2.imshow(
        "VIRTUALFIT V3.4 - Body Measurements",
        preview
    )

    print("Press Q to close.")

    while True:
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
