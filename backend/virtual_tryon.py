
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
