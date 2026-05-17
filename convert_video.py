import cv2

input_path = "data/shiotsu.mp4"
output_path = "data/shiotsu_fixed.mp4"

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"Error: Cannot open {input_path}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Use MP4V codec (compatible)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")

cap.release()
out.release()
print(f"Done. Saved {output_path} with {frame_count} frames.")