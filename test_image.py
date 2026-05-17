import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

img = cv2.imread("person.jpg")
if img is None:
    print("Error: Could not load person.jpg. Make sure the file exists.")
    exit()

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    res = pose.process(rgb)
    if res.pose_landmarks:
        print("✅ Pose detected in static image! MediaPipe works.")
        mp_draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite("pose_output.jpg", img)
        print("Annotated image saved as pose_output.jpg")
    else:
        print("❌ No pose detected in static image. MediaPipe is broken.")