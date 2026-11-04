
import cv2
import mediapipe as mp
import numpy as np
import math

print("AI VIRTUAL TRY-ON STARTED")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not detected!")
    exit()

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

shirt_options = {
    1: ("BLUE T-SHIRT", (255, 100, 30)),
    2: ("BLACK T-SHIRT", (40, 40, 40)),
    3: ("RED T-SHIRT", (30, 30, 220))
}

selected = 1


def draw_shirt(frame, landmarks, shirt_color):

    h, w = frame.shape[:2]

    def point(index):
        p = landmarks[index]
        return (int(p.x * w), int(p.y * h))

    # Body landmarks
    ls = point(11)
    rs = point(12)
    lh = point(23)
    rh = point(24)

    # Check visibility
    for index in [11, 12, 23, 24]:
        if landmarks[index].visibility < 0.5:
            return frame

    shoulder_width = math.dist(ls, rs)

    if shoulder_width < 30:
        return frame

    center_x = (ls[0] + rs[0]) // 2
    top_y = (ls[1] + rs[1]) // 2

    neck = int(shoulder_width * 0.13)
    sleeve = int(shoulder_width * 0.32)

    # T-shirt shape
    shirt = np.array([
        [center_x - neck, top_y + int(shoulder_width * 0.05)],
        [ls[0] - int(shoulder_width * 0.08), ls[1]],
        [ls[0] - sleeve, ls[1] + int(shoulder_width * 0.22)],
        [ls[0] - int(shoulder_width * 0.15), ls[1] + int(shoulder_width * 0.38)],
        [lh[0] - int(shoulder_width * 0.12), lh[1]],
        [rh[0] + int(shoulder_width * 0.12), rh[1]],
        [rs[0] + int(shoulder_width * 0.15), rs[1] + int(shoulder_width * 0.38)],
        [rs[0] + sleeve, rs[1] + int(shoulder_width * 0.22)],
        [rs[0] + int(shoulder_width * 0.08), rs[1]],
        [center_x + neck, top_y + int(shoulder_width * 0.05)]
    ], dtype=np.int32)

    overlay = frame.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(overlay, [shirt], shirt_color)
    cv2.fillPoly(mask, [shirt], 255)

    # Neck opening
    cv2.ellipse(
        mask,
        (center_x, top_y + int(shoulder_width * 0.07)),
        (neck, int(shoulder_width * 0.09)),
        0, 0, 360, 0, -1
    )

    alpha = 0.65

    blended = cv2.addWeighted(
        frame, 1 - alpha,
        overlay, alpha, 0
    )

    frame[mask > 0] = blended[mask > 0]

    return frame


print("Webcam connected!")
print("Press 1, 2, 3 to change clothes.")
print("Press Q to quit.")

while True:

    success, frame = cap.read()

    if not success:
        print("Cannot read webcam!")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:

        frame = draw_shirt(
            frame,
            results.pose_landmarks.landmark,
            shirt_options[selected][1]
        )

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.putText(
        frame,
        "VIRTUAL TRY-ON",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "OUTFIT: " + shirt_options[selected][0],
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press 1 / 2 / 3 to change outfit",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI VIRTUAL TRY-ON", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        selected = 1

    elif key == ord('2'):
        selected = 2

    elif key == ord('3'):
        selected = 3

    elif key == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()

print("Virtual Try-On closed successfully!")
