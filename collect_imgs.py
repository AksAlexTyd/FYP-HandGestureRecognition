import os
import cv2
import numpy as np


def collect_data(labels, dataset_size=1000, data_dir='./data'):
    """
    Collects dataset images for each label in the given labels list.
    
    Parameters:
        labels (list of str): List of labels for which data should be collected.
        dataset_size (int): Number of images to collect per label. Default is 1000.
        data_dir (str): Directory to save the dataset. Default is './data'.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    cap = cv2.VideoCapture(0)

    # Set the window to fullscreen
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Notify user about the current label collection
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, f'Collecting data for label "{label}"',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, 'Press "Q" to start!',
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        # Data collection loop
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture video frame.")
                break

            # Save the frame with the current counter as the file name
            cv2.imwrite(os.path.join(label_dir, f'{counter}.jpg'), frame)
            counter += 1

            # Display the frame with a dynamic message
            cv2.putText(frame, f'Capturing images: {counter}/{dataset_size}',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()


# # Example usage
# if __name__ == "__main__":
#     labels = ["Label1", "Label2", "Label3"]
#     collect_data(labels, dataset_size=50, data_dir='./data')
