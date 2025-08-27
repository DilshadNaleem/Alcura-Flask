import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class MedicineClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medicine Classifier Pro")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')

        # Set confidence threshold (adjust as needed)
        self.confidence_threshold = 0.7

        # Load the trained model and drug info
        try:
            self.model = load_model("medicine_classifier_model.h5")
            self.class_names = sorted(os.listdir("Pil-DataSet/train"))  # Get class names from dataset folder
            self.drug_info = pd.read_csv("Pil-DataSet/drug_information.csv")

            # Ensure 'unknown' class exists in drug_info
            if 'unknown' not in self.drug_info['class_name'].values:
                unknown_data = {
                    'class_name': 'unknown',
                    'dosage': 'N/A',
                    'use': 'Not a medicine',
                    'price': 'N/A',
                    'side_effects': 'N/A',
                    'dosage_form': 'N/A',
                    'Scientific_Name' : 'N/A',
                    'max_dose': 'N/A',
                    'administration': 'N/A',
                    'indications': 'N/A',
                    'precautions': 'N/A',
                    'serious_effects': 'N/A',
                    'contraindications': 'N/A',
                    'Source_of_information' : 'N/A'
                }
                self.drug_info = pd.concat([self.drug_info, pd.DataFrame([unknown_data])], ignore_index=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model or drug info: {str(e)}")
            self.root.destroy()
            return

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Custom style
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 20, 'bold'))
        style.configure('Result.TLabel', font=('Arial', 14, 'bold'))
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.configure('Info.TLabel', font=('Arial', 11), background='#ffffff', padding=5)

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Medicine Classifier Pro", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Left panel (image and controls)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, padx=10, sticky=tk.N)

        # Image display
        self.image_label = ttk.Label(left_frame)
        self.image_label.pack(pady=10)

        # Select image button
        select_btn = ttk.Button(left_frame, text="Select Medicine Image", command=self.select_image)
        select_btn.pack(pady=5, fill=tk.X)

        # Classify button
        classify_btn = ttk.Button(left_frame, text="Classify Medicine", command=self.classify_image)
        classify_btn.pack(pady=5, fill=tk.X)

        # Right panel (results and info)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, padx=10, sticky=tk.NSEW)

        # Result display
        self.result_label = ttk.Label(right_frame, text="No medicine classified yet", style='Result.TLabel')
        self.result_label.pack(pady=(0, 10), anchor=tk.W)

        # Confidence display
        self.confidence_label = ttk.Label(right_frame, text="")
        self.confidence_label.pack(anchor=tk.W)

        # Info notebook (tabs for different info)
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Create tabs
        self.details_tab = ttk.Frame(self.notebook)
        self.dosage_tab = ttk.Frame(self.notebook)
        self.usage_tab = ttk.Frame(self.notebook)
        self.side_effects_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.details_tab, text="Details")
        self.notebook.add(self.dosage_tab, text="Dosage")
        self.notebook.add(self.usage_tab, text="Usage")
        self.notebook.add(self.side_effects_tab, text="Side Effects")

        # Initialize info labels
        self.detail_labels = {}
        for tab in [self.details_tab, self.dosage_tab, self.usage_tab, self.side_effects_tab]:
            for i in range(5):  # Create empty labels that we'll update later
                label = ttk.Label(tab, text="", style='Info.TLabel', wraplength=300)
                label.pack(fill=tk.X, padx=5, pady=2, anchor=tk.W)
                self.detail_labels[f"{tab}_{i}"] = label

        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(10, 0))

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Medicine Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if file_path:
            try:
                img = Image.open(file_path)
                img.thumbnail((400, 400))  # Resize for display
                photo = ImageTk.PhotoImage(img)

                self.image_label.config(image=photo)
                self.image_label.image = photo  # Keep reference
                self.current_image_path = file_path

                # Clear previous results
                self.result_label.config(text="No medicine classified yet")
                self.confidence_label.config(text="")
                self.clear_drug_info()
                self.status_bar.config(text=f"Loaded: {os.path.basename(file_path)}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_bar.config(text="Error loading image")

    def clear_drug_info(self):
        """Clear all drug information displays"""
        for label in self.detail_labels.values():
            label.config(text="")

    def classify_image(self):
        if not hasattr(self, 'current_image_path'):
            messagebox.showwarning("Warning", "Please select an image first")
            return

        try:
            # Load and preprocess the image
            img = image.load_img(self.current_image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index]

            # Get class name
            predicted_class = self.class_names[predicted_class_index]

            # Check confidence threshold
            if confidence < self.confidence_threshold:
                predicted_class = "unknown"
                self.result_label.config(text="Not a recognized medicine")
                self.confidence_label.config(text=f"Confidence: {confidence:.2%} (too low)")
            else:
                self.result_label.config(text=f"Predicted Medicine: {predicted_class}")
                self.confidence_label.config(text=f"Confidence: {confidence:.2%}")

            self.status_bar.config(text=f"Classified as {predicted_class} with {confidence:.2%} confidence")

            # Show drug information
            self.display_drug_info(predicted_class)

        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.status_bar.config(text="Classification error")

    def display_drug_info(self, drug_name):
        """Display all information about the identified drug"""
        try:
            # Check if drug exists in the info dataframe
            if drug_name not in self.drug_info['class_name'].values:
                drug_name = "unknown"

            drug_data = self.drug_info[self.drug_info['class_name'] == drug_name].iloc[0]

            # Details tab
            self.detail_labels[f"{self.details_tab}_0"].config(text=f"Name: {drug_data['class_name']}")
            self.detail_labels[f"{self.details_tab}_1"].config(
                text=f"Dosage Form: {drug_data.get('dosage', 'N/A')}")
            self.detail_labels[f"{self.details_tab}_2"].config(text=f"Price: {drug_data.get('price', 'N/A')}")
            self.detail_labels[f"{self.details_tab}_3"].config(text=f"Scientic Name: {drug_data.get('Scientific_Name', 'N/A')}")
            self.detail_labels[f"{self.details_tab}_4"].config(text=f"Source of Information: {drug_data.get('Source_of_information', 'N/A')}")

            # Dosage tab
            self.detail_labels[f"{self.dosage_tab}_0"].config(text=f"Standard Dosage: {drug_data.get('dosage', 'N/A')}")
            self.detail_labels[f"{self.dosage_tab}_1"].config(
                text=f"Max Daily Dose: {drug_data.get('max_dose', 'N/A')}")
            self.detail_labels[f"{self.dosage_tab}_2"].config(
                text=f"Administration: {drug_data.get('administration', 'N/A')}")

            # Usage tab
            self.detail_labels[f"{self.usage_tab}_0"].config(text=f"Primary Use: {drug_data.get('use', 'N/A')}")
            self.detail_labels[f"{self.usage_tab}_1"].config(
                text=f"Indications: {drug_data.get('indications', 'N/A')}")
            self.detail_labels[f"{self.usage_tab}_2"].config(
                text=f"Precautions: {drug_data.get('precautions', 'N/A')}")

            # Side Effects tab
            self.detail_labels[f"{self.side_effects_tab}_0"].config(
                text=f"Common Side Effects: {drug_data.get('side_effects', 'N/A')}")
            self.detail_labels[f"{self.side_effects_tab}_1"].config(
                text=f"Serious Side Effects: {drug_data.get('serious_effects', 'N/A')}")
            self.detail_labels[f"{self.side_effects_tab}_2"].config(
                text=f"Contraindications: {drug_data.get('contraindications', 'N/A')}")

        except Exception as e:
            messagebox.showwarning("Information", f"Could not load drug information: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MedicineClassifierApp(root)
    root.mainloop()