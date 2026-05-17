import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker.task"

# If model not present, download it
import os
import urllib.request
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Downloaded.")

# Initialize landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# Read first frame from video
cap = cv2.VideoCapture("data/shiotsu.mp4")
ret, frame = cap.read()
cap.release()
if not ret:
    print("Failed to read video frame")
    exit()

# Convert to RGB and create MediaPipe Image
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

# Detect
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    detection_result = landmarker.detect(mp_image)
    if detection_result.pose_landmarks:
        print(f"✅ Pose detected! Number of landmarks: {len(detection_result.pose_landmarks[0])}")
        # Save the annotated image to verify
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        pose = detection_result.pose_landmarks[0]
        for lm in pose:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 3, (0,255,0), -1)
        cv2.imwrite("test_output.jpg", annotated)
        print("Saved annotated frame as test_output.jpg")
    else:
        print("❌ No pose detected")