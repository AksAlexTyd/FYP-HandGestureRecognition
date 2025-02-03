import os
import pickle
import mediapipe as mp
import cv2


def create_dataset(data_dir, output_file):
    """
    Creates a dataset from hand landmarks in the specified folder and saves it as a .pickle file.
    
    Parameters:
        data_dir (str): Path to the dataset folder. Each subfolder should represent a label.
        output_file (str): Path to the output .pickle file.
    
    Returns:
        None
    """
    # Initialize MediaPipe Hands and dataset storage
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

    data = []
    labels = []

    # Iterate through each label directory
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):  # Skip non-directory entries
            continue
        
        print(f"Processing label: {label}")
        
        for img_file in os.listdir(label_dir):
            data_aux = []
            x_ = []
            y_ = []

            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read image {img_path}. Skipping.")
                continue

            # Convert image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    # Normalize landmarks to the bounding box
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Append processed data and labels
                data.append(data_aux)
                labels.append(label)

    # Save processed data to a .pickle file
    with open(output_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    
    print(f"Dataset created and saved to {output_file}")


# # # Example Usage
# if __name__ == "__main__":
#     create_dataset(data_dir='./data/left', output_file="dataLeft.pickle")
#     create_dataset(data_dir='./data/right', output_file="dataRight.pickle")
