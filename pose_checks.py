from config import THRESHOLD_ANGLE, THRESHOLD_ALIGNMENT


def check_straight(angle):
    return abs(angle - 180) < THRESHOLD_ANGLE

def check_arm_vertical(shoulder, wrist):
    return wrist[1] < shoulder[1]

def check_body_alignment(shoulder, hip, ankle):
    return (
        abs(shoulder[0] - hip[0]) < THRESHOLD_ALIGNMENT and
        abs(hip[0] - ankle[0]) < THRESHOLD_ALIGNMENT
    )

def evaluate_pose(angles, landmarks):
    feedback = []

    if not check_straight(angles["left_elbow"]):
        feedback.append("Straighten left arm")
    if not check_straight(angles["right_elbow"]):
        feedback.append("Straighten right arm")

    if not check_straight(angles["left_knee"]):
        feedback.append("Straighten left knee")
    if not check_straight(angles["right_knee"]):
        feedback.append("Straighten right knee")

    if not check_arm_vertical(landmarks["left_shoulder"], landmarks["left_wrist"]):
        feedback.append("Raise left arm")
    if not check_arm_vertical(landmarks["right_shoulder"], landmarks["right_wrist"]):
        feedback.append("Raise right arm")

    if not check_body_alignment(
        landmarks["left_shoulder"],
        landmarks["left_hip"],
        landmarks["left_ankle"]
    ):
        feedback.append("Keep body straight")

    if len(feedback) == 0:
        return ["Perfect pose! Great Job!"]

    return feedback

def calculate_accuracy(angles):
    scores = []

    for joint, angle in angles.items():
        score = max(0, 100 - abs(angle - 180))
        scores.append(score)

    return int(sum(scores) / len(scores))

def is_pose_ready(angles, landmarks):
    if not check_arm_vertical(landmarks["left_shoulder"], landmarks["left_wrist"]):
        return False
    if not check_arm_vertical(landmarks["right_shoulder"], landmarks["right_wrist"]):
        return False
    if not check_straight(angles["left_knee"]):
        return False
    if not check_straight(angles["right_knee"]):
        return False

    return True
