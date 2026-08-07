import cv2
import mediapipe as mp
from angle_utils import calculate_angle
from pose_checks import evaluate_pose, calculate_accuracy, is_pose_ready
from config import JOINTS

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)


def get_point(landmarks, index):
    return [landmarks[index].x, landmarks[index].y]


# 🔥 UI FUNCTION (FINAL FRONTEND)
def draw_ui(frame, accuracy, feedback):
    height, width, _ = frame.shape

    # Top bar
    cv2.rectangle(frame, (0, 0), (width, 60), (40, 40, 40), -1)
    cv2.putText(frame, "PoseMate Trainer",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    # Side panel
    cv2.rectangle(frame, (0, 60), (250, height), (30, 30, 30), -1)

    # Accuracy section
    if accuracy is not None:
        cv2.putText(frame, f"{accuracy}%",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3)

        bar_width = int((accuracy / 100) * 200)
        cv2.rectangle(frame, (20, 180), (220, 210), (100, 100, 100), -1)
        cv2.rectangle(frame, (20, 180), (20 + bar_width, 210), (0, 255, 0), -1)

    else:
        cv2.putText(frame, "Not Ready",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    # Center feedback
    y = 300
    for f in feedback:
        text_size = cv2.getTextSize(f, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = int((width - text_size[0]) / 2)

        if "Perfect" in f:
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)

        cv2.putText(frame, f, (text_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        y += 40


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Landmarks
        landmarks = {
            "left_shoulder": get_point(lm, 11),
            "right_shoulder": get_point(lm, 12),
            "left_elbow": get_point(lm, 13),
            "right_elbow": get_point(lm, 14),
            "left_wrist": get_point(lm, 15),
            "right_wrist": get_point(lm, 16),
            "left_hip": get_point(lm, 23),
            "right_hip": get_point(lm, 24),
            "left_knee": get_point(lm, 25),
            "right_knee": get_point(lm, 26),
            "left_ankle": get_point(lm, 27),
            "right_ankle": get_point(lm, 28),
        }

        # Angles
        angles = {}
        for joint, (a, b, c) in JOINTS.items():
            angles[joint] = calculate_angle(
                get_point(lm, a),
                get_point(lm, b),
                get_point(lm, c)
            )

        # Evaluate
        feedback = evaluate_pose(angles, landmarks)

        # Accuracy check
        if is_pose_ready(angles, landmarks):
            accuracy = calculate_accuracy(angles)
        else:
            accuracy = None

        # Draw UI
        draw_ui(frame, accuracy, feedback)

    cv2.imshow("PoseMate - Mountain Pose", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

#venv\Scripts\activate

#python -m streamlit run app.py