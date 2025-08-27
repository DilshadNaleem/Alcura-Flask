import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import pandas as pd
import os

# Load model and disease info
model = tf.keras.models.load_model("disease_classifier_model.h5")
disease_df = pd.read_csv("Disease/disease_information.csv")

# Constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
CONFIDENCE_THRESHOLD = 0.7

# Set class names (assumes generator was saved during training)
# If not, manually define your class names in the same order as training
class_names = sorted(os.listdir("Disease/train"))
class_indices = {i: name for i, name in enumerate(class_names)}

# Prediction function
def predict_image(img_path):
    try:
        img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]
        predicted_class = class_indices[class_idx]

        if confidence < CONFIDENCE_THRESHOLD or predicted_class.lower() == "unknown":
            return "Not recognized", confidence, None

        info = disease_df[disease_df['Disease'].str.lower() == predicted_class.lower()]
        return predicted_class, confidence, info.iloc[0] if not info.empty else None
    except Exception as e:
        return "Error", 0, str(e)

# GUI
class DiseaseClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Disease Classifier")

        self.label = tk.Label(root, text="Upload an image to classify", font=("Helvetica", 14))
        self.label.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=5)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.result_text = tk.Text(root, height=15, width=70, wrap="word")
        self.result_text.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpeg *.jpg *.png")])
        if not file_path:
            return

        self.display_image(file_path)
        self.classify_image(file_path)

    def display_image(self, path):
        img = Image.open(path)
        img = img.resize((200, 200))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def classify_image(self, path):
        self.result_text.delete(1.0, tk.END)
        predicted_class, confidence, info = predict_image(path)

        if predicted_class == "Error":
            messagebox.showerror("Prediction Error", f"Error occurred: {info}")
            return

        result = f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}\n\n"

        if info is not None:
            for col in ['Description', 'symptoms', 'treatment', 'medications', 'first_aid_advice']:
                result += f"{col.capitalize()}: {info.get(col, 'N/A')}\n"

        elif predicted_class == "Not recognized":
            result += "This image does not appear to be a recognized disease."

        self.result_text.insert(tk.END, result)

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = DiseaseClassifierApp(root)
    root.mainloop()
