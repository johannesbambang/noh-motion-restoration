import cv2
import mediapipe as mp
import json
import os

MODEL_PATH = "pose_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture("data/shiotsu.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
max_frames = fps * 5  # first 5 seconds
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("results/quick_tracked.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

landmarks_data = {}
frame_idx = 0

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(frame_idx * 1000 / fps)
        detection = landmarker.detect_for_video(mp_image, timestamp)

        frame_landmarks = []
        if detection.pose_landmarks:
            pose = detection.pose_landmarks[0]
            for lm in pose:
                frame_landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z, "visibility": getattr(lm, 'visibility', 1.0)})
            # Draw skeleton (simplified)
            h, w = frame.shape[:2]
            for lm in pose:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0,255,0), -1)
            connections = [(11,12),(11,13),(13,15),(12,14),(14,16),(23,24),(23,25),(25,27),(24,26),(26,28),(11,23),(12,24)]
            for s,e in connections:
                if s < len(pose) and e < len(pose):
                    p1 = (int(pose[s].x * w), int(pose[s].y * h))
                    p2 = (int(pose[e].x * w), int(pose[e].y * h))
                    cv2.line(frame, p1, p2, (255,0,0), 2)
        else:
            print(f"Frame {frame_idx}: No pose")
        landmarks_data[frame_idx] = frame_landmarks
        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")

cap.release()
out.release()

with open("results/quick_landmarks.json", "w") as f:
    json.dump(landmarks_data, f, indent=2)

print(f"Saved JSON with {len(landmarks_data)} frames to results/quick_landmarks.json")
print("Saved video to results/quick_tracked.mp4")