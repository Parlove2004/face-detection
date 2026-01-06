import cv2
import os
import time
from datetime import datetime

# Create folder
os.makedirs("captured_frames", exist_ok=True)

#  Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Cannot access webcam")
    exit()

#  Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#  Motion detection setup
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

last_saved_time = time.time()

print(" Smart Webcam Monitoring Started | Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #  Face Detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #  Motion Detection
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    motion_pixels = cv2.countNonZero(thresh)

    motion_detected = motion_pixels > 5000
    if motion_detected:
        cv2.putText(frame, "Motion Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #  Save frame every 2 seconds (only if motion OR face detected)
    current_time = time.time()
    if current_time - last_saved_time >= 2:
        if motion_detected or len(faces) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_frames/frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
            last_saved_time = current_time

    prev_gray = gray

    cv2.imshow("Smart Webcam Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(" Monitoring stopped")
