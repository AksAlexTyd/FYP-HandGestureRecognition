import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
import pickle

# Global flag to stop both camera threads when 'q' is pressed
stop_threads = False  

def hand_sign_recognition(model_left, model_right, camera_id):
    """
    Perform real-time hand sign recognition using the provided models.
    Use left-hand predictions for keyboard input and right-hand predictions for mouse pointer actions.

    Args:
        model_left: Trained model for recognizing left-hand signs.
        model_right: Trained model for recognizing right-hand signs.
        camera_id: Camera ID (0 for first camera, 1 for second camera)
    """
    global stop_threads  # Access the global stop flag

    # Open video capture for the specified camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Unable to access camera {camera_id}.")
        return

    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)


    # Create a named window
    window_name = f'Hand Sign Recognition - Camera {camera_id}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    try:
        while not stop_threads:  # Stop when the global flag is set
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Failed to grab frame from camera {camera_id}. Exiting...")
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

                    # Make predictions based on hand type
                    if hand_label == "Left":
                        predicted_label = model_left.predict([np.asarray(data_aux)])[0]
                        print(f"Left Hand Prediction (Camera {camera_id}): {predicted_label}")
                        pyautogui.press(str(predicted_label))

                    elif hand_label == "Right" and camera_id==1:
                        predicted_label = model_right.predict([np.asarray(data_aux)])[0]
                        print(f"Right Hand Prediction (Camera {camera_id}): {predicted_label}")

                        # Perform mouse actions based on right-hand model output
                        if predicted_label == "Pointer":
                            screen_width, screen_height = pyautogui.size()
                            mouse_x = int(np.mean(x_) * screen_width)
                            mouse_y = int(np.mean(y_) * screen_height)
                            pyautogui.moveTo(mouse_x, mouse_y)

                        elif predicted_label == "Left Click":
            
                            pyautogui.click()
                            print(f"Left Click Detected (Camera {camera_id})")


                        elif predicted_label == "Right Click":
                           
                            pyautogui.rightClick()
                            print(f"Right Click Detected (Camera {camera_id})")

            # Display the frame
            cv2.imshow(window_name, frame)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested. Stopping all cameras...")
                stop_threads = True
                break  # Exit loop for this camera

    finally:
        # Release resources
        cap.release()
        cv2.destroyWindow(window_name)  # Close only the window for this camera
        print(f"Camera {camera_id} released and window closed.")

# Create and run threads for parallel execution
def start_parallel_hand_sign_recognition(model_left_1, model_right_1, model_left_2, model_right_2):
    global stop_threads
    stop_threads = False  # Reset the flag before starting

    thread_1 = threading.Thread(target=hand_sign_recognition, args=(model_left_1, model_right_1, 1))  # PC Camera
    thread_2 = threading.Thread(target=hand_sign_recognition, args=(model_left_2, model_right_2, 0))  # USB Webcam

    thread_1.start()
    thread_2.start()

    # Wait for both threads to complete
    thread_1.join()
    thread_2.join()

    print("Both cameras stopped successfully.")

# # Load the trained models
# model_left_dict = pickle.load(open("./modelLeft.p", "rb"))
# model_right_dict = pickle.load(open("./modelRight.p", "rb"))
# model_left_1 = model_left_dict["model"]
# model_right_1 = model_right_dict["model"]
# model_left_2 = model_left_dict["model"]
# model_right_2 = model_right_dict["model"]

# # Start the parallel recognition process
# start_parallel_hand_sign_recognition(model_left_1, model_right_1, model_left_2, model_right_2)
