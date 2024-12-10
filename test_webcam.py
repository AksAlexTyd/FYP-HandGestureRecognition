import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the models
model_left_dict = pickle.load(open('./modelLeft.p', 'rb'))
model_right_dict = pickle.load(open('./modelRight.p', 'rb'))
model_left = model_left_dict['model']
model_right = model_right_dict['model']

# Open video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

# Label dictionaries
labels_left = {5: '5',6: '6', 7: '7', 8: '8'}
labels_right = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            data_aux = []
            x_ = []
            y_ = []

            # Get handedness label (Left or Right)
            hand_label = handedness.classification[0].label

            # Draw landmarks and collect data
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Make predictions based on hand type
            if hand_label == "Left":  # This will now correspond to the user's physical left hand
                prediction = model_left.predict([np.asarray(data_aux)])
                predicted_character = labels_left[int(prediction[0])]
            elif hand_label == "Right":  # This will now correspond to the user's physical right hand
                prediction = model_right.predict([np.asarray(data_aux)])
                predicted_character = labels_right[int(prediction[0])]

            print(f"Hand: {hand_label}, Predicted label: {predicted_character}")

            # Draw prediction on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
