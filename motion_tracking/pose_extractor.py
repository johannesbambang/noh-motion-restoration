#!/usr/bin/env python3
"""
Pose extraction for Noh performance using MediaPipe Tasks API.
Works with mediapipe 0.10.30+ (no 'solutions' module).
"""

import cv2
import mediapipe as mp
import json
import argparse
import os
import urllib.request

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_PATH = "pose_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

def extract_pose(video_path, output_json_path, output_video_path, display=False):
    download_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarks_data = {}
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB and create MediaPipe Image using top-level mp.Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(frame_idx * 1000 / fps)
            detection = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_landmarks = []
            if detection.pose_landmarks:
                pose = detection.pose_landmarks[0]
                for lm in pose:
                    frame_landmarks.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": getattr(lm, 'visibility', 1.0)
                    })

                # Draw landmarks and connections (simple OpenCV drawing)
                h, w = frame.shape[:2]
                for lm in pose:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                # Simplified skeleton connections (based on MediaPipe indices)
                connections = [
                    (11,12), (11,13), (13,15), (12,14), (14,16),
                    (23,24), (23,25), (25,27), (24,26), (26,28),
                    (11,23), (12,24), (15,21), (16,22)
                ]
                for s, e in connections:
                    if s < len(pose) and e < len(pose):
                        p1 = (int(pose[s].x * w), int(pose[s].y * h))
                        p2 = (int(pose[e].x * w), int(pose[e].y * h))
                        cv2.line(frame, p1, p2, (255, 0, 0), 2)
            else:
                print(f"Frame {frame_idx}: No pose detected")

            landmarks_data[frame_idx] = frame_landmarks
            out.write(frame)

            if display:
                cv2.imshow('Pose Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames")

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

    with open(output_json_path, 'w') as f:
        json.dump(landmarks_data, f, indent=2)

    print(f"Saved: {output_json_path}")
    print(f"Saved: {output_video_path}")
    print(f"Total frames: {frame_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_json", default="results/landmarks.json")
    parser.add_argument("--output_video", default="results/pose_tracked.mp4")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)

    extract_pose(args.input, args.output_json, args.output_video, args.display)