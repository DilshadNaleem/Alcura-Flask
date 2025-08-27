import face_recognition
import numpy as np
import cv2
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def image_to_encoding(image_path):
    try:
        image = face_recognition.load_image_file(image_path)

        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print("No faces detected in image")
            return None

        # OpenCV eye detection
        image_bgr = cv2.imread(image_path)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        print(f"Eyes detected: {len(eyes)}")

        if len(eyes) < 1:
            print("No open eyes detected")
            return "closed_eyes"

        encoding = face_recognition.face_encodings(image)[0]
        return encoding.tolist()

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

@app.route('/register_face', methods=['POST'])
def register_face():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "message": "No image provided"})

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"success": False, "message": "No selected file"})

        temp_path = f"temp_{image_file.filename}"
        image_file.save(temp_path)

        encoding = image_to_encoding(temp_path)
        os.remove(temp_path)

        if encoding == "closed_eyes":
            return jsonify({"success": False, "message": "Eyes are closed. Please open your eyes and try again."})

        if encoding is None:
            return jsonify({"success": False, "message": "No face detected"})

        return jsonify({
            "success": True,
            "encoding": encoding
        })

    except Exception as e:
        print(f"Error in register_face: {str(e)}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "message": "No image provided", "match": False})

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"success": False, "message": "No selected file", "match": False})

        stored_encoding = request.form.get('stored_encoding')
        if not stored_encoding:
            return jsonify({"success": False, "message": "No stored encoding provided", "match": False})

        # Save uploaded image
        os.makedirs("temp_uploads", exist_ok=True)
        temp_path = os.path.join("temp_uploads", f"verify_temp_{image_file.filename}")
        image_file.save(temp_path)

        # Eye detection using only haarcascade_eye.xml
        image_bgr = cv2.imread(temp_path)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Require at least 2 eyes detected
        if len(eyes) < 2:
            os.remove(temp_path)
            return jsonify({
                "success": False,
                "message": "Eyes not clearly detected. Please open your eyes and look directly at the camera.",
                "match": False
            })

        # Proceed with face encoding
        current_encoding = image_to_encoding(temp_path)
        os.remove(temp_path)

        if current_encoding is None:
            return jsonify({"success": False, "message": "No face detected", "match": False})
        if current_encoding == "closed_eyes":
            return jsonify({"success": False, "message": "Eyes appear to be closed", "match": False})

        # Compare with stored encoding
        try:
            stored_array = np.array(json.loads(stored_encoding))
        except json.JSONDecodeError:
            return jsonify({"success": False, "message": "Invalid stored encoding format", "match": False})

        match_result = face_recognition.compare_faces(
            [stored_array],
            np.array(current_encoding),
            tolerance=0.4
        )[0]

        return jsonify({
            "success": True,
            "message": "Face verification completed",
            "match": bool(match_result)
        })

    except Exception as e:
        print(f"Error in verify_face: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Verification error: {str(e)}",
            "match": False
        })




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)