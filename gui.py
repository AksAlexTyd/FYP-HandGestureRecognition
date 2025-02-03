import tkinter as tk
from tkinter import ttk, messagebox
import threading
import pickle
from collect_imgs import collect_data
from create_dataset import create_dataset
from train_classifier import train_classifier
from testModels import hand_sign_recognition
from testCameras import  start_parallel_hand_sign_recognition

def validate_inputs(left_text_boxes, right_dropdowns):
    for i, text_box in enumerate(left_text_boxes):
        value = text_box.get().strip()
        if len(value) != 1:
            messagebox.showwarning("Validation Error", f"Class Label {i+1} must contain exactly one character.")
            return False

    selected_actions = [dropdown.get() for dropdown in right_dropdowns if dropdown.get() != "Choose Pointer Action"]
    if len(selected_actions) < 2 or len(selected_actions) != len(set(selected_actions)):
        messagebox.showwarning("Validation Error", "Select at least two unique pointer actions.")
        return False

    return True

def show_loading_window(root, message):
    loading_window = tk.Toplevel(root)
    loading_window.title("Please Wait")
    loading_window.geometry("300x100")
    tk.Label(loading_window, text=message).pack(pady=10)
    progress_bar = ttk.Progressbar(loading_window, mode="indeterminate")
    progress_bar.pack(pady=10)
    progress_bar.start()
    return loading_window

def collectData(valid_labels, selected_actions, root):
    loading_window = show_loading_window(root, "Collecting Data...")
    
    def process_data():
        collect_data(labels=valid_labels, dataset_size=100, data_dir="./data/left")
        collect_data(labels=selected_actions, dataset_size=100, data_dir="./data/right")
        loading_window.destroy()
        createDataset(root)
    
    threading.Thread(target=process_data).start()

def createDataset(root):
    loading_window = show_loading_window(root, "Creating Dataset...")
    
    def process_dataset():
        create_dataset(data_dir="./data/left", output_file="dataLeft.pickle")
        create_dataset(data_dir="./data/right", output_file="dataRight.pickle")
        loading_window.destroy()
        trainClassifier(root)
    
    threading.Thread(target=process_dataset).start()

def trainClassifier(root):
    loading_window = show_loading_window(root, "Training Classifier...")
    
    def process_training():
        train_classifier(data_path="./dataLeft.pickle", model_path="modelLeft.p", test_size=0.2)
        train_classifier(data_path="./dataRight.pickle", model_path="modelRight.p", test_size=0.2)
        loading_window.destroy()
        messagebox.showinfo("Success", "Training completed successfully!")
    
    threading.Thread(target=process_training).start()



import os
from tkinter import messagebox

def remove_model():
    selected_option = remove_model_dropdown.get()
    models_deleted = []

    if selected_option in ["Left Model", "Both Models"]:
        if os.path.exists("modelLeft.p"):
            os.remove("modelLeft.p")
            models_deleted.append("Left Model")
        else:
            messagebox.showwarning("Warning", "Left Model does not exist!")

    if selected_option in ["Right Model", "Both Models"]:
        if os.path.exists("modelRight.p"):
            os.remove("modelRight.p")
            models_deleted.append("Right Model")
        else:
            messagebox.showwarning("Warning", "Right Model does not exist!")

    if models_deleted:
        messagebox.showinfo("Success", f"Deleted: {', '.join(models_deleted)}")
    else:
        messagebox.showwarning("Nothing to Delete", "No models found to delete!")




def create_gui():
    root = tk.Tk()
    root.title("Hand Sign Recognition")
    
    left_frame = tk.Frame(root, padx=10, pady=10, relief="groove", borderwidth=2)
    left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew",columnspan=2)
    
    left_rows = []
    
    def add_label_input():
        row = len(left_rows) + 1
        label = tk.Label(left_frame, text=f"Class Label {row}")
        label.grid(row=row, column=0, padx=5, pady=5)
        text_box = tk.Entry(left_frame)
        text_box.grid(row=row, column=1, padx=5, pady=5)
        clear_button = ttk.Button(left_frame, text="X", command=lambda: remove_row([label, text_box, clear_button]))
        clear_button.grid(row=row, column=2, padx=5, pady=5)
        left_rows.append([label, text_box, clear_button])
    
    def remove_row(row_widgets):
        for widget in row_widgets:
            widget.destroy()
        left_rows.remove(row_widgets)
    
    for _ in range(5):
        add_label_input()
    
    add_label_button = ttk.Button(left_frame, text="Add Label", command=add_label_input)
    add_label_button.grid(row=999, column=0, columnspan=3, pady=10)
    
    right_frame = tk.Frame(root, padx=10, pady=10, relief="groove", borderwidth=2)
    right_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew",columnspan=2)
    
    right_dropdowns = []
    for i in range(3):
        tk.Label(right_frame, text=f"Class Label {i+1}").grid(row=i, column=0, padx=5, pady=5)
        dropdown = ttk.Combobox(right_frame, values=["Pointer", "Left Click", "Right Click"], state="readonly")
        dropdown.set("Choose Pointer Action")
        dropdown.grid(row=i, column=1, padx=5, pady=5)
        right_dropdowns.append(dropdown)
    
    def start_training():
        left_inputs = [row[1] for row in left_rows]
        if validate_inputs(left_inputs, right_dropdowns):
            valid_labels = [text_box.get().strip() for text_box in left_inputs]
            selected_actions = [dropdown.get() for dropdown in right_dropdowns if dropdown.get() != "Choose Pointer Action"]
            collectData(valid_labels, selected_actions, root)
    
    ttk.Button(root, text="Start Training", command=start_training).grid(row=1, column=0, pady=10)
    
    # Test Model with Camera Mode Dropdown
    def test_model_with_camera_mode():
        selected_camera_mode = camera_mode_dropdown.get()
        if selected_camera_mode == "One Camera":
            use_two_cameras = False
        elif selected_camera_mode == "Two Cameras":
            use_two_cameras = True
        else:
            messagebox.showwarning("Camera Mode Error", "Please select a valid camera mode.")
            return

        test_trained_models(root, use_two_cameras)
     # Model Management Section
    camera_frame = tk.Frame(root, padx=10, pady=10, relief="groove", borderwidth=2)
    camera_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")
    # Camera Mode Dropdown for video test
    camera_mode_label = tk.Label(camera_frame, text="Select Camera Mode:")
    camera_mode_label.grid(row=0, column=0, pady=5)
    
    camera_mode_dropdown = ttk.Combobox(camera_frame, values=["One Camera", "Two Cameras"], state="readonly")
    camera_mode_dropdown.set("One Camera")
    camera_mode_dropdown.grid(row=0, column=1, pady=5)

    test_model_button = ttk.Button(camera_frame, text="Test Model", command=test_model_with_camera_mode)
    test_model_button.grid(row=0, column=2, pady=10)
    
   # Model Management Section
    model_frame = tk.Frame(root, padx=10, pady=10, relief="groove", borderwidth=2)
    model_frame.grid(row=3, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")

    tk.Label(model_frame, text="Remove Model:").grid(row=0, column=0, padx=5, pady=5)
    global remove_model_dropdown
    remove_model_dropdown = ttk.Combobox(model_frame, values=["Left Model", "Right Model", "Both Models"], state="readonly")
    remove_model_dropdown.set("Left Model")
    remove_model_dropdown.grid(row=0, column=1, padx=5, pady=5)

    remove_model_button = ttk.Button(model_frame, text="Remove Model", command=remove_model)
    remove_model_button.grid(row=0, column=2, padx=5, pady=5)
    
       
    ttk.Button(root, text="Exit", command=root.destroy).grid(row=4, column=3, pady=10)

    root.mainloop()

def test_trained_models(root, use_two_cameras):
    try:
        if use_two_cameras:
                # Call the function with appropriate model arguments
            # Replace `model_left_1`, `model_right_1`, `model_left_2`, and `model_right_2` with your actual model objects
            model_left_dict = pickle.load(open("./modelLeft.p", "rb"))
            model_right_dict = pickle.load(open("./modelRight.p", "rb"))
            model_left_1 = model_left_dict["model"]
            model_right_1 = model_right_dict["model"]
            model_left_2 = model_left_dict["model"]
            model_right_2 = model_right_dict["model"]
                    
            start_parallel_hand_sign_recognition(model_left_1, model_right_1, model_left_2, model_right_2)

        else:       
            model_left_dict = pickle.load(open("./modelLeft.p", "rb"))
            model_right_dict = pickle.load(open("./modelRight.p", "rb"))
            model_left = model_left_dict["model"]
            model_right = model_right_dict["model"]
            
            # hand_sign_recognition(model_left, model_right, use_two_cameras)
            hand_sign_recognition(model_left, model_right)
        
       
    
    except FileNotFoundError as e:
        messagebox.showerror("Error", f"Model file not found: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")


create_gui()
