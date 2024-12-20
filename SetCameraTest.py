import cv2

# Set the webcam index here (e.g., 0, 1, etc.)
default_webcam_index = 1
cap = cv2.VideoCapture(default_webcam_index)

if not cap.isOpened():
    print("Error: Could not access the selected webcam.")
    exit()

# Display webcam feed
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
