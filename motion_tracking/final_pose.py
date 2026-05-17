import cv2
import mediapipe as mp
import json

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("data/shiotsu.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("results/final_tracked.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

landmarks = {}
frame = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, img = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        frame_data = []
        if res.pose_landmarks:
            mp_draw.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for lm in res.pose_landmarks.landmark:
                frame_data.append({"x": lm.x, "y": lm.y, "z": lm.z, "vis": lm.visibility})
        else:
            print(f"Frame {frame}: no pose")
        landmarks[frame] = frame_data
        out.write(img)
        frame += 1
        if frame % 100 == 0:
            print(f"Processed {frame} frames")

cap.release()
out.release()
with open("results/final_landmarks.json", "w") as f:
    json.dump(landmarks, f, indent=2)
print(f"Done. Total frames: {frame}")