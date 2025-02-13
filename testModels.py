import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

def hand_sign_recognition(model_left, model_right):
    """
    Perform real-time hand sign recognition using the provided models.
    Use left-hand predictions for keyboard input and right-hand predictions for mouse pointer actions.

    Args:
        model_left: Trained model for recognizing left-hand signs.
        model_right: Trained model for recognizing right-hand signs.
    """
    # Handle key inputs
    lastKeyInput = ''

    # Open video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

    last_click_time = 0
    double_click_interval = 0.5  # 500 ms for double click detection
    is_dragging = False  # Flag to detect dragging state

    # Create a named window
    window_name = 'Hand Sign Recognition'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Set the window to be resizable
    cv2.resizeWindow(window_name, 800, 600)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame. Exiting...")
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            print(lastKeyInput)

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

                    # Make predictions based on hand type
                    if hand_label == "Left":
                        predicted_label = model_left.predict([np.asarray(data_aux)])[0]
                        print(f"Left Hand Prediction: {predicted_label}")
                        previous_label = lastKeyInput
                        if previous_label != predicted_label:
                            pyautogui.keyUp(str(previous_label))
                        lastKeyInput = predicted_label
                        pyautogui.keyDown(str(predicted_label))

                    elif hand_label == "Right":
                        predicted_label = model_right.predict([np.asarray(data_aux)])[0]
                        print(f"Right Hand Prediction: {predicted_label}")

                        # Perform mouse actions based on right-hand model output
                        if predicted_label == "Pointer":
                            # Get the screen width and height
                            screen_width, screen_height = pyautogui.size()

                            # Scale the hand's mean position to the screen dimensions
                            mouse_x = int(np.mean(x_) * screen_width)
                            mouse_y = int(np.mean(y_) * screen_height)

                            # Move the mouse to the calculated position
                            pyautogui.moveTo(mouse_x, mouse_y)

                        elif predicted_label == "Left Click":
                            current_time = time.time()
                            if current_time - last_click_time <= double_click_interval:
                                # Wait for camera feed to stabilize the "Select" prediction
                                time.sleep(0.2)  # Slight pause to allow stabilization
                                pyautogui.doubleClick()
                                print("Double Click Detected")
                            else:
                                # Wait for camera stabilization before single click
                                time.sleep(0.2)
                                pyautogui.click()
                                print("Left Click Detected")
                            last_click_time = current_time

                        elif predicted_label == "Right Click":
                            # Wait briefly to confirm the gesture is consistent
                            time.sleep(0.2)
                            pyautogui.rightClick()
                            print("Right Click Detected")
            else:
                # print("not detected")
                pyautogui.keyUp(str(lastKeyInput))

            # Display the frame
            cv2.imshow(window_name, frame)

            # Check window status
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed. Exiting...")
                break

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested by user.")
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
