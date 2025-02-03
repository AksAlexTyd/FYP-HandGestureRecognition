# Hand Gesture Recognition System

This project is a hand gesture recognition system that captures images, processes hand landmarks, trains a classifier, and allows real-time gesture recognition for controlling the computer.

## Project Overview

The system consists of multiple components:

- **Image Collection** (`collect_imgs.py`): Captures images from a webcam for different hand gestures.
- **Dataset Creation** (`create_dataset.py`): Extracts hand landmarks and stores them in a dataset.
- **Model Training** (`train_classifier.py`): Trains a Random Forest model for gesture classification.
- **Graphical User Interface (GUI)** (`gui.py`): Provides an interactive way to collect data, train models, and test recognition.
- **Real-time Testing** (`testModels.py`, `testCameras.py`): Runs the trained model on live camera input and performs actions like clicking or moving the mouse.

## Installation

1. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe numpy pickle-mixin scikit-learn pyautogui
   ```
2. Ensure you have a working webcam for image collection and testing.

## Usage

### 1. Collect Data

Run the following command to collect hand gesture images:

```bash
python collect_imgs.py
```

Follow the on-screen instructions to capture images for different gestures.

### 2. Create Dataset

Once images are collected, extract hand landmarks and create a dataset:

```bash
python create_dataset.py
```

This will generate `.pickle` files containing hand landmark data.

### 3. Train Model

Train the Random Forest classifier using:

```bash
python train_classifier.py
```

This will save trained models for gesture recognition.

### 4. Run the GUI

To use a graphical interface for managing the process:

```bash
python gui.py
```

The GUI allows you to collect data, train models, and test recognition easily.

### 5. Test Gesture Recognition

For single-camera recognition:

```bash
python testModels.py
```

For dual-camera setup:

```bash
python testCameras.py
```

This will use the trained models to recognize gestures in real time.

## Features

- Captures and processes hand images
- Uses MediaPipe for hand landmark detection
- Trains a Random Forest classifier for gesture recognition
- Provides a GUI for easy interaction
- Controls mouse and keyboard using hand gestures

## Notes

- Ensure proper lighting conditions for better recognition.
- The dataset size impacts accuracy; collect enough images for each gesture.
- Press `q` to exit real-time recognition.

