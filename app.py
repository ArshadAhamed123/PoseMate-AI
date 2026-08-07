import streamlit as st
import cv2
import mediapipe as mp
from angle_utils import calculate_angle
from pose_checks import evaluate_pose, calculate_accuracy, is_pose_ready
from config import JOINTS

st.set_page_config(layout="wide")
st.title("🧠 PoseMate Trainer")

if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = ""
if "accuracy" not in st.session_state:
    st.session_state.accuracy = 0

run = st.checkbox("Start Camera")

POSES = {
    "Mountain Pose": {
        "image": "pose.png",
        "steps": [
            "Stand straight with feet together",
            "Raise both arms upward",
            "Join both hands",
            "Keep arms straight",
            "Align body vertically",
            "Hold the posture"
        ]
    }
}

pose_name = st.selectbox(
    "Select Pose",
    ["-- Select a Pose --"] + list(POSES.keys())
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def get_point(landmarks, index):
    return [landmarks[index].x, landmarks[index].y]

if pose_name != "-- Select a Pose --":

    pose_data = POSES[pose_name]

    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()

    with col2:
        st.subheader("📌 Reference Pose")

        img_col, text_col = st.columns([1, 1])

        with img_col:
            try:
                st.image(pose_data["image"], width=220)
            except:
                st.warning("Image not found")

        with text_col:
            st.subheader("📝 Instructions")
            for step in pose_data["steps"]:
                st.write(f"• {step}")

        accuracy_box = st.empty()
        feedback_box = st.empty()

    cap = cv2.VideoCapture(0)

    if run:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not working")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = cv2.flip(image, 1)

            results = pose.process(image)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

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

                angles = {}
                for joint, (a, b, c) in JOINTS.items():
                    angles[joint] = calculate_angle(
                        get_point(lm, a),
                        get_point(lm, b),
                        get_point(lm, c)
                    )

                feedback_list = evaluate_pose(angles, landmarks)

                if feedback_list:
                    main_feedback = feedback_list[0]

                    if main_feedback != st.session_state.last_feedback:
                        st.session_state.last_feedback = main_feedback

                        if "good" in main_feedback.lower() or "perfect" in main_feedback.lower():
                            feedback_box.success(main_feedback)
                        else:
                            feedback_box.warning(main_feedback)

                if is_pose_ready(angles, landmarks):
                    accuracy = calculate_accuracy(angles)

                    if accuracy != st.session_state.accuracy:
                        st.session_state.accuracy = accuracy
                        accuracy_box.metric("Accuracy", f"{accuracy}%")
                else:
                    accuracy_box.warning("Get into position")

            frame_placeholder.image(image, channels="RGB")

    cap.release()

if pose_name != "-- Select a Pose --":

    st.markdown("---")
    
    with st.expander("💡 See Benefits"):
        benefits = {
            "Mountain Pose": [
                "Improves posture and balance",
                "Strengthens thighs, knees, and ankles",
                "Increases body awareness",
                "Enhances focus and stability",
                "Helps in alignment correction"
            ]
        }

        for b in benefits.get(pose_name, []):
            st.write(f"✅ {b}")
