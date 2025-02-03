import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_classifier(data_path='./dataLeft.pickle', model_path='modelLeft.p', test_size=0.2):
    """
    Trains a Random Forest Classifier on the provided dataset and evaluates its performance.
    
    Parameters:
        data_path (str): Path to the dataset pickle file.
        model_path (str): Path to save the trained model.
        test_size (float): Fraction of the dataset to use for testing. Default is 0.2 (20%).
    """
    try:
        # Load dataset
        data_dict = pickle.load(open(data_path, 'rb'))
        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, shuffle=True, stratify=labels
        )

        # Initialize the model
        n_estimators = 100
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Output model details
        print("Number of decision trees in the Random Forest:", len(model.estimators_))

        # Evaluate the model
        y_predict = model.predict(X_test)
        score = accuracy_score(y_predict, y_test)
        print(f'{score * 100:.2f}% of samples were classified correctly!')

        # Save the trained model
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model}, f)
        print(f'Model saved to {model_path}')
    except Exception as e:
        print(f"Error occurred: {e}")


# Example GUI to call the function from another frame
# class MainApp(tk.Tk):
#     def __init__(self):
#         super().__init__()

#         self.title("Main GUI")
#         self.geometry("400x200")

#         # Add a button to start training
#         train_button = tk.Button(self, text="Train Model", command=self.start_training)
#         train_button.pack(pady=50)

#     def start_training(self):
#         # Call the function to start training directly (no thread)
#         train_classifier('./dataLeft.pickle', 'modelLeft.p', test_size=0.2)


# # Run the GUI application
# if __name__ == "__main__":
#     app = MainApp()
#     app.mainloop()
