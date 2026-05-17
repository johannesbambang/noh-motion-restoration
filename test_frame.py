import cv2

cap = cv2.VideoCapture("data/shiotsu.mp4")
ret, frame = cap.read()
if ret:
    cv2.imwrite("test_frame.jpg", frame)
    print("Saved test_frame.jpg – open it to see the first frame.")
else:
    print("Could not read frame.")
cap.release()