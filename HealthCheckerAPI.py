from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and drug info
try:
    model = load_model("medicine_classifier_model.h5")
    class_names = sorted(os.listdir("Pil-DataSet/train"))
    drug_info = pd.read_csv("Pil-DataSet/drug_information.csv")


    # Ensure 'unknown' class exists in drug_info
    if 'unknown' not in drug_info['class_name'].values:
        unknown_data = {
            'class_name': 'unknown',
            'dosage': 'unknown',
            'use': 'Not a medicine',
            'price': 'unknown',
            'side_effects': 'unknown',
            'dosage_form': 'unknown',
            'Scientific_Name': 'unknown',
            'max_dose': 'unknown',
            'administration': 'unknown',
            'indications': 'unknown',
            'precautions': 'unknown',
            'serious_effects': 'unknown',
            'contraindications': 'unknown',
            'Source_of_information': 'unknown'
        }
        drug_info = pd.concat([drug_info, pd.DataFrame([unknown_data])], ignore_index=True)

except Exception as e:
    print(f"Failed to load model or drug info: {str(e)}")
    exit(1)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_image(image_path):
    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]

        # Get class name
        predicted_class = class_names[predicted_class_index]

        # Check confidence threshold
        if confidence < 0.7:  # confidence threshold
            predicted_class = "unknown"

        # Get drug information
        drug_data = drug_info[drug_info['class_name'] == predicted_class]
        if drug_data.empty:
            drug_data = drug_info[drug_info['class_name'] == "unknown"]

        drug_info_dict = drug_data.iloc[0].to_dict()

        return {
            "status": "success",
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "drug_info": drug_info_dict
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.route('/classify', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = classify_image(filepath)

        # Clean up - remove the uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(result)

    return jsonify({"status": "error", "message": "Invalid file type"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,  debug= True)