import socket
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Socket setup
HOST = '127.0.0.1'  # Localhost (or Unity's IP address if running remotely)
PORT = 65432        # Port number to communicate with Unity
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)

# Load gesture recognition model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: "WAVE", 1: "FIST", 2: "LEFT_SWIPE", 3: "RIGHT_SWIPE", 4: "TWO_HANDS_UP"}

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Predict gesture
            prediction = model.predict([np.asarray(data_aux)])
            gesture = labels_dict[int(prediction[0])]

            # Send gesture to Unity
            sock.sendall(gesture.encode('utf-8'))

            # Draw the prediction on the frame
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
