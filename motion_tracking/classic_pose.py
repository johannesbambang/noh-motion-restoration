import cv2
import mediapipe as mp
import json

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Open video
cap = cv2.VideoCapture("data/shiotsu.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("results/classic_tracked.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

landmarks_data = {}
frame_idx = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        
        frame_landmarks = []
        if result.pose_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            for lm in result.pose_landmarks.landmark:
                frame_landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
        else:
            print(f"Frame {frame_idx}: No pose")
        
        landmarks_data[frame_idx] = frame_landmarks
        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")

cap.release()
out.release()

with open("results/classic_landmarks.json", "w") as f:
    json.dump(landmarks_data, f, indent=2)

print(f"Done. Processed {frame_idx} frames.")
print("Saved: results/classic_tracked.mp4 and results/classic_landmarks.json")