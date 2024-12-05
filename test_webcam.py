# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np

# # Load the model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Open video capture
# cap = cv2.VideoCapture(0)

# # Initialize Mediapipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

# # Label dictionary
# labels_dict = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}

# while True:
#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame. Exiting...")
#         break

#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     if results.multi_hand_landmarks:
#         # Process only the first detected hand to avoid breakdowns
#         hand_landmarks = results.multi_hand_landmarks[0]
#         mp_drawing.draw_landmarks(
#             frame,  # image to draw
#             hand_landmarks,  # model output
#             mp_hands.HAND_CONNECTIONS,  # hand connections
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())

#         for i in range(len(hand_landmarks.landmark)):
#             x = hand_landmarks.landmark[i].x
#             y = hand_landmarks.landmark[i].y
#             x_.append(x)
#             y_.append(y)

#         for i in range(len(hand_landmarks.landmark)):
#             x = hand_landmarks.landmark[i].x
#             y = hand_landmarks.landmark[i].y
#             data_aux.append(x - min(x_))
#             data_aux.append(y - min(y_))

#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10
#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10

#         # Make predictions
#         prediction = model.predict([np.asarray(data_aux)])
#         predicted_character = labels_dict[int(prediction[0])]

#         # Draw prediction on frame
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)

#     # Display the frame
#     cv2.imshow('frame', frame)

#     # Quit on 'q' key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Quitting...")
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

# Label dictionary
labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',5:'5'}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            # Draw landmarks and collect data
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
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

            # Make predictions
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            
            print("Predicte label: ",predicted_character)

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