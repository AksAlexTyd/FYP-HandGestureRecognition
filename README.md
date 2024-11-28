# FYP-HandGestureRecognition


Hand Gesture Recognition System
===============================

Overview
--------
This project implements a hand gesture recognition system using computer vision and machine learning.
It captures hand gesture images, extracts features using MediaPipe, trains a classifier, and predicts 
gestures in real time using webcam input.

Files
-----
1. collect_imgs.py
   - **Purpose**: Captures images of hand gestures using a webcam and saves them for training.
   - **Key Features**:
     - Organizes data into folders based on gesture classes.
     - Captures 100 images per gesture class by default.
     - Displays a live preview and waits for user confirmation before starting capture.

2. create_dataset.py
   - **Purpose**: Processes captured images to extract hand landmark data using MediaPipe.
   - **Key Features**:
     - Extracts and normalizes x and y coordinates of hand landmarks.
     - Stores processed data and labels into a data.pickle file for training.

3. train_classifier.py
   - **Purpose**: Trains a Random Forest classifier on the processed dataset.
   - **Key Features**:
     - Splits data into training and testing sets.
     - Evaluates the model’s performance and prints accuracy.
     - Saves the trained model to model.p.

4. test_webcam.py
   - **Purpose**: Uses the trained model to predict hand gestures in real time.
   - **Key Features**:
     - Captures live video feed via webcam.
     - Detects hand landmarks, normalizes features, and predicts the gesture class.
     - Displays the prediction and bounding box on the video feed.

--------------------------------------------------------------------------

Installation
------------
1. **Prerequisites**:
   - Python 3.8 or higher
   - Libraries: OpenCV, MediaPipe, Scikit-learn, Matplotlib, NumPy

   Install dependencies using:
   ```
   pip install opencv-python mediapipe scikit-learn matplotlib numpy
   ```

2. **Setup**:
   - Create a 'data' directory in the root folder for saving images.
   - Run `collect_imgs.py` to capture gesture images.

Usage
-----
1. **Collect Data**:
   ```
   python collect_imgs.py
   ```
   Follow on-screen instructions to capture images for each gesture class.

2. **Create Dataset**:
   ```
   python create_dataset.py
   ```
   This processes captured images and prepares data for training.

3. **Train Classifier**:
   ```
   python train_classifier.py
   ```
   This trains a Random Forest model and evaluates its accuracy.

4. **Test in Real-Time**:
   ```
   python test_webcam.py
   ```
   This uses the webcam feed to recognize and display hand gestures in real time.

Notes
-----
- Ensure sufficient lighting during image capture for better recognition.
- Add more gesture classes by updating the scripts and capturing new data.
- Test on various hand sizes and skin tones to improve robustness.
