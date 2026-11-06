import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not detected!")

mp_selfie = mp.solutions.selfie_segmentation

with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:

    print("PERSON SEGMENTATION ACTIVE!")
    print("Press Q to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = segmenter.process(rgb)

        mask = result.segmentation_mask
        mask = (mask > 0.5).astype(np.uint8) * 255

        segmented = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("Original Webcam", frame)
        cv2.imshow("Person Segmentation", segmented)
        cv2.imshow("Segmentation Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
