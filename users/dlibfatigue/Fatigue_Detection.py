from scipy.spatial import distance
from imutils import face_utils
from datetime import datetime
import imutils
import dlib
import cv2
import winsound

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    return (A + B) / (2.0 * C)

# Thresholds
thresh = 0.25
frame_check = 20

# Face detector and landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Webcam
cap = cv2.VideoCapture(0)

flag = 0
blink_count = 0
screenshot_taken = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    subjects = detect(gray, 0)

    status = "AWAKE"

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1

            if flag >= frame_check:
                status = "DROWSY"

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "WAKE UP!",
                    (200, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # Continuous alarm
                winsound.Beep(2500, 200)

                # Save screenshot once
                if not screenshot_taken:
                    cv2.imwrite("drowsy_driver.jpg", frame)
                    screenshot_taken = True

        else:
            if flag >= 3:
                blink_count += 1

            flag = 0
            screenshot_taken = False

        # EAR Display
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (400, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # Status Display
    cv2.putText(
        frame,
        f"Status: {status}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Blink Counter
    cv2.putText(
        frame,
        f"Blinks: {blink_count}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Time Display
    current_time = datetime.now().strftime("%H:%M:%S")

    cv2.putText(
        frame,
        current_time,
        (400, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()