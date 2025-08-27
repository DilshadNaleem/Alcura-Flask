from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'Disease_Uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and data
try:
    model = load_model("disease_classifier_model.h5")
    class_names = sorted(os.listdir("Disease/train"))
    disease_info = pd.read_csv("Disease/disease_information.csv")

    # Ensure 'unknown' entry exists
    if 'unknown' not in disease_info['Disease'].values:
        unknown_data = {
            'Disease': 'unknown',
            'Description': 'unknown',
            'symptoms': 'unknown',
            'cause': 'unknown',
            'side_effects': 'unknown',
            'treatment': 'unknown',
            'medications': 'unknown',
            'prevention': 'unknown',
            'severity': 'unknown',
            'risk_factors': 'unknown',
            'is_contagious': 'No',
            'common_age_group': 'unknown',
            'duration': 'unknown',
            'first_aid_advice': 'unknown',
            'Source_of_information': 'unknown',
            'scientific_name': 'unknown'
        }
        disease_info = pd.concat([disease_info, pd.DataFrame([unknown_data])], ignore_index=True)

except Exception as e:
    print(f"Failed to load model or data: {e}")
    exit(1)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions_result = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions_result[0])
        confidence = predictions_result[0][predicted_class_index]
        predicted_class = class_names[predicted_class_index]

        if confidence < 0.7:
            predicted_class = "unknown"

        disease_data = disease_info[disease_info["Disease"] == predicted_class]
        if disease_data.empty:
            disease_data = disease_info[disease_info["Disease"] == "unknown"]

        return {
            "status": "success",
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "disease_info": disease_data.iloc[0].to_dict()
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during classification: {e}"
        }


@app.route('/DiseaseClassify', methods=['POST'])
def upload_file():
    # Debug print to check if file is received
    print("Request.files keys:", request.files.keys())

    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = classify_image(filepath)

        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Warning: Failed to delete uploaded file. Error: {e}")

        return jsonify(result)

    return jsonify({"status": "error", "message": "Invalid file type. Allowed types: png, jpg, jpeg"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
