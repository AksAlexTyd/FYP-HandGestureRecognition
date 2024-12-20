
import cv2

index = 0
print("Checking available webcams...")
while True:
    cap = cv2.VideoCapture(index)
    if not cap.read()[0]:
        print(f"No webcam found at index {index}.")
        cap.release()
        break
    else:
        print(f"Webcam found at index {index}.")
        cap.release()
    index += 1
