import base64
import os
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import pandas as pd
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import io
import matplotlib.pyplot as plt
import traceback
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg to prevent GUI issues

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/api/*": {"origins" : "http:/localhost:8081"}})
# Configuration
UPLOAD_FOLDER = 'uploads'
TRAIN_FOLDER = 'Disease/train'
VAL_FOLDER = 'Disease/validation'
DISEASE_INFO_FILE = "Disease/disease_information.csv"
MODEL_FILE = "disease_classifier_model.h5"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(VAL_FOLDER, exist_ok=True)

# Model parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 4
CONFIDENCE_THRESHOLD = 0.7

# Initialize disease dataframe
disease_df = pd.DataFrame(columns=[
    'Disease', 'Description', 'symptoms', 'cause', 'side_effects',
    'treatment', 'medications', 'prevention', 'severity', 'risk_factors',
    'is_contagious', 'common_age_group', 'duration', 'first_aid_advice',
    'Source_of_information', 'scientific_name'
])
if os.path.exists(DISEASE_INFO_FILE):
    disease_df = pd.read_csv(DISEASE_INFO_FILE)

# Initialize model
model = None
if os.path.exists(MODEL_FILE):
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        print("Pre-trained model loaded successfully!")
    except:
        print("Could not load pre-trained model. Train a new one.")


def build_model(num_classes):
    """Build and compile the MobileNetV2 model"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


@app.route('/api/diseases', methods=['GET']) #View only disease names
def get_diseases():
    """Get list of all diseases"""
    return jsonify({
        'diseases': disease_df['Disease'].unique().tolist()
    })



@app.route('/api/AllDiseases', methods=['GET']) # View all diseases with descriptions
def get_all_diseases():
    """Get complete list of all diseases (full details)"""
    all_diseases = disease_df.to_dict('records')  # Converts DataFrame to list of dictionaries
    return jsonify(all_diseases)

@app.route('/api/disease', methods=['POST']) #add Disease to dataset
def add_disease():
    """Add or update disease information with images"""
    try:
        # Get form data
        disease_data = request.form.get('diseaseData')
        if not disease_data:
            return jsonify({'error': 'Disease data is required'}), 400

        # Parse disease data (assuming it's sent as JSON string)
        import json
        disease_info = json.loads(disease_data)

        if 'Disease' not in disease_info:
            return jsonify({'error': 'Disease name is required'}), 400

        disease_name = disease_info['Disease']

        # Use original disease name for folder creation (only replace spaces with single space)
        folder_name = ' '.join(disease_name.split())

        # Process disease info
        existing_index = disease_df.index[disease_df['Disease'] == disease_name].tolist()

        new_row = {
            'Disease': '', 'Description': '', 'symptoms': '', 'cause': '', 'side_effects': '',
            'treatment': '', 'medications': '', 'prevention': '', 'severity': '', 'risk_factors': '',
            'is_contagious': '', 'common_age_group': '', 'duration': '', 'first_aid_advice': '',
            'Source_of_information': '', 'scientific_name': ''
        }
        new_row.update({k: v for k, v in disease_info.items() if k in new_row})

        if existing_index:
            disease_df.loc[existing_index[0]] = new_row
        else:
            disease_df.loc[len(disease_df)] = new_row

        disease_df.to_csv(DISEASE_INFO_FILE, index=False)

        # Create folders if they don't exist
        train_folder = os.path.join(TRAIN_FOLDER, folder_name)
        val_folder = os.path.join(VAL_FOLDER, folder_name)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        # Save training images (still use secure_filename for actual files)
        train_count = 0
        if 'trainImages' in request.files:
            for file in request.files.getlist('trainImages'):
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    # Ensure filename is safe and has an image extension
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    try:
                        file.save(os.path.join(train_folder, filename))
                        train_count += 1
                    except Exception as e:
                        app.logger.error(f"Error saving training image {filename}: {str(e)}")
                        continue

        # Save validation images
        val_count = 0
        if 'valImages' in request.files:
            for file in request.files.getlist('valImages'):
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    # Ensure filename is safe and has an image extension
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    try:
                        file.save(os.path.join(val_folder, filename))
                        val_count += 1
                    except Exception as e:
                        app.logger.error(f"Error saving validation image {filename}: {str(e)}")
                        continue

        return jsonify({
            'status': 'success',
            'message': f'Disease information saved with {train_count} training images and {val_count} validation images',
            'folder_name': folder_name
        }), 200

    except Exception as e:
        app.logger.error(f"Error in add_disease: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/disease/<disease_name>', methods=['DELETE']) #Delete diseases from system
def delete_disease(disease_name):
    """Delete a disease and all its images"""
    # Delete folders
    train_folder = os.path.join(TRAIN_FOLDER, disease_name)
    val_folder = os.path.join(VAL_FOLDER, disease_name)

    try:
        # Delete image folders
        if os.path.exists(train_folder):
            shutil.rmtree(train_folder)
        if os.path.exists(val_folder):
            shutil.rmtree(val_folder)

        # Remove disease from dataframe - with case-insensitive and whitespace handling
        global disease_df
        original_count = len(disease_df)

        # Convert both to lowercase and strip whitespace for comparison
        disease_df = disease_df[~disease_df['Disease'].str.strip().str.lower().eq(disease_name.strip().lower())]

        new_count = len(disease_df)

        if original_count == new_count:
            return jsonify({'warning': f'Disease "{disease_name}" not found in dataframe'}), 404

        # Save updated dataframe to CSV
        disease_df.to_csv(DISEASE_INFO_FILE, index=False)

        return jsonify({'message': f'Deleted all images and information for {disease_name}'}) , 200
    except Exception as e:
        return jsonify({'error': f'Could not delete: {str(e)}'}), 500



@app.route('/api/train', methods=['POST'])

def train_model():
    """Train the disease classification model"""
    try:
        # Enable eager execution for compatibility
        tf.config.run_functions_eagerly(True)

        # Check if we have enough data for training
        train_classes = [d for d in os.listdir(TRAIN_FOLDER) if os.path.isdir(os.path.join(TRAIN_FOLDER, d))]
        val_classes = [d for d in os.listdir(VAL_FOLDER) if os.path.isdir(os.path.join(VAL_FOLDER, d))]

        if not train_classes:
            return jsonify({'error': 'No training data found! Please add images first.'}), 400

        if not val_classes:
            return jsonify({'error': 'No validation data found! Please add images first.'}), 400

        # Create data generators with proper initialization
        class CustomImageDataGenerator(ImageDataGenerator):
            def _init_(self, **kwargs):
                super()._init_(**kwargs)  # Fixes the PyDataset warning

        train_datagen = CustomImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = CustomImageDataGenerator(rescale=1.0 / 255)

        # Training generator
        train_generator = train_datagen.flow_from_directory(
            TRAIN_FOLDER,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
        )

        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            VAL_FOLDER,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
        )

        # Build or load model with proper handling
        global model
        needs_rebuild = (
                model is None or
                model.output_shape[1] != train_generator.num_classes or
                not hasattr(model, 'optimizer') or
                not isinstance(model.optimizer, tf.keras.optimizers.Optimizer)
        )

        if needs_rebuild:
            model = build_model(train_generator.num_classes)
            print("Built new model with", train_generator.num_classes, "classes")
        else:
            # Recompile existing model with fresh optimizer
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("Recompiled existing model")

        # Train the model
        epochs = int(request.form.get('epochs', 10))
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            steps_per_epoch=max(1, len(train_generator)),
            validation_steps=max(1, len(val_generator)),
            verbose=1
        ).history

        # Save the model
        model.save(MODEL_FILE)
        print("Model saved successfully")

        # Convert history values to lists (they might be numpy arrays)
        history = {k: [float(v) for v in vals] for k, vals in history.items()}

        # Generate training charts
        plt.figure(figsize=(12, 5))

        # Accuracy chart
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        # Loss chart
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        # Save chart to bytes
        chart_bytes = io.BytesIO()
        plt.savefig(chart_bytes, format='png')
        plt.close()
        chart_bytes.seek(0)
        chart_base64 = base64.b64encode(chart_bytes.read()).decode('utf-8')

        return jsonify({
            'message': 'Model training completed successfully!',
            'history': history,
            'chart': f"data:image/png;base64,{chart_base64}"
        })

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return jsonify({
            'error': f'Training failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/predict', methods=['POST'])  # Predict Disease
def predict():
    """Predict disease from an uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    if model is None:
        return jsonify({'error': 'No trained model found. Please train a model first.'}), 400

    file = request.files['image']

    try:
        # Load and preprocess image
        img = Image.open(file.stream).resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx])

        # Get class names from training data
        train_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
            TRAIN_FOLDER,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
        )
        class_names = {v: k for k, v in train_generator.class_indices.items()}
        predicted_class = class_names[class_idx]

        # Prepare response
        response = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_recognized': confidence >= CONFIDENCE_THRESHOLD
        }

        # Add disease information if confidence is high enough
        if confidence >= CONFIDENCE_THRESHOLD:
            disease_info = disease_df[disease_df['Disease'] == predicted_class]
            if not disease_info.empty:
                response['disease_info'] = disease_info.iloc[0].to_dict()

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Could not process image: {str(e)}'}), 500


@app.route('/api/validation_results', methods=['GET']) # Get Validations
def get_validation_results():
    """Get validation results with images and confidence scores"""
    if model is None:
        return jsonify({'error': 'No trained model found. Please train a model first.'}), 400

    # Create data generator for validation set
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_generator = val_datagen.flow_from_directory(
        VAL_FOLDER,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=1,  # Process one image at a time
        class_mode='categorical',
        shuffle=False
    )

    # Get class names
    class_names = {v: k for k, v in val_generator.class_indices.items()}
    results = []

    # Process all validation images (limit to 20 for demo)
    for i in range(min(20, len(val_generator))):
        # Get image and true label
        img, true_label = val_generator[i]
        true_class_idx = np.argmax(true_label[0])
        true_class = class_names[true_class_idx]

        # Make prediction
        prediction = model.predict(img)
        pred_class_idx = np.argmax(prediction[0])
        pred_class = class_names[pred_class_idx]
        confidence = float(prediction[0][pred_class_idx])

        # Convert image to bytes for response
        img_bytes = io.BytesIO()
        plt.imshow(img[0])
        plt.axis('off')
        plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img_bytes.seek(0)

        results.append({
            'true_class': true_class,
            'predicted_class': pred_class,
            'confidence': confidence,
            'is_correct': true_class == pred_class,
            'image': f"data:image/png;base64,{base64.b64encode(img_bytes.read()).decode('utf-8')}"
        })

    return jsonify(results)


@app.route('/api/validation_results/search', methods=['GET']) #Search in validation results
def search_validation_results():
    """Search/filter validation results"""
    if model is None:
        return jsonify({'error': 'No trained model found. Please train a model first.'}), 400

    # Get query parameters
    class_name = request.args.get('class', '').lower()
    is_correct = request.args.get('is_correct', None)

    # Get all validation results (reusing the existing function)
    all_results = get_validation_results().get_json()

    # Filter results
    filtered_results = []
    for result in all_results:
        # Filter by class name (case insensitive)
        if class_name and (class_name not in result['true_class'].lower() and
                           class_name not in result['predicted_class'].lower()):
            continue

        # Filter by correctness
        if is_correct is not None:
            expected_correct = is_correct.lower() == 'true'
            if result['is_correct'] != expected_correct:
                continue

        filtered_results.append(result)

    return jsonify(filtered_results)

@app.route('/api/images/<disease_name>', methods=['POST'])  #Add Images to the folder
def upload_images(disease_name):
    """Upload images for a specific disease"""
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    dataset_type = request.form.get('dataset_type', 'train')
    if dataset_type not in ['train', 'validation']:
        return jsonify({'error': 'Invalid dataset type. Use "train" or "validation"'}), 400

    target_folder = TRAIN_FOLDER if dataset_type == 'train' else VAL_FOLDER
    disease_folder = os.path.join(target_folder, disease_name)
    os.makedirs(disease_folder, exist_ok=True)

    files = request.files.getlist('images')
    saved_files = []

    for file in files:
        try:
            # Verify image
            img = Image.open(file.stream)
            img.verify()

            # Save to target folder
            filename = secure_filename(file.filename)
            filepath = os.path.join(disease_folder, filename)
            file.seek(0)  # Reset file pointer after verify
            file.save(filepath)
            saved_files.append(filename)
        except Exception as e:
            continue  # Skip invalid images

    return jsonify({
        'message': f'Uploaded {len(saved_files)} images for {disease_name}',
        'saved_files': saved_files
    })

@app.route('/api/disease/<old_name>', methods=['POST']) #Edit disease name
def edit_disease(old_name):
    """Edit disease information and synchronize folders (without sanitizing spaces)"""
    try:
        # Get all fields from the request
        data = request.json
        new_name = data.get('new_name')

        if not new_name:
            return jsonify({'error': 'new_name is required'}), 400

        # Check if disease exists (exact match, case-sensitive)
        if old_name not in disease_df['Disease'].values:
            return jsonify({'error': f'Disease "{old_name}" not found'}), 404

        # 1. Update all fields in the DataFrame (keep original formatting)
        idx = disease_df[disease_df['Disease'] == old_name].index[0]

        # Update each field that was provided in the request
        for column in disease_df.columns:
            if column in data and column != 'Disease':  # Disease name is handled separately
                disease_df.at[idx, column] = data[column]

        # Update the disease name (keeping original spaces)
        disease_df.at[idx, 'Disease'] = new_name

        # Save the updated DataFrame (preserves original formatting)
        disease_df.to_csv(DISEASE_INFO_FILE, index=False)

        # 2. Rename folders (WITHOUT replacing spaces with underscores)
        if old_name != new_name:
            old_train = os.path.join(TRAIN_FOLDER, old_name)  # No sanitization
            new_train = os.path.join(TRAIN_FOLDER, new_name)  # No sanitization
            old_val = os.path.join(VAL_FOLDER, old_name)      # No sanitization
            new_val = os.path.join(VAL_FOLDER, new_name)      # No sanitization

            if os.path.exists(old_train):
                os.rename(old_train, new_train)
            if os.path.exists(old_val):
                os.rename(old_val, new_val)

        return jsonify({
            'message': f'Updated disease "{old_name}"',
            'old_folder_name': old_name if old_name != new_name else None,
            'new_folder_name': new_name if old_name != new_name else None,
            'updated_fields': list(data.keys())
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


import pandas as pd


@app.route('/api/disease/<disease_name>', methods=['GET']) # Search All fields by name
def get_disease_info(disease_name):
    """Get disease info by directly searching the CSV file"""
    try:
        # Read the CSV file fresh every time (to get latest updates)
        current_df = pd.read_csv(DISEASE_INFO_FILE)

        # Search for exact match (case-sensitive)
        disease_data = current_df[current_df['Disease'].str.strip().str.lower() == disease_name.strip().lower()]

        if disease_data.empty:
            return jsonify({'error': f'Disease "{disease_name}" not found'}), 404

        # Convert the matching row to a dictionary
        result = disease_data.iloc[0].to_dict()
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

from urllib.parse import unquote
def possible_folder_names(name):
    """Generate possible folder name variants to match real folders."""
    decoded = unquote(name.replace('+', ' '))
    base_variants = set([
        decoded,
        decoded.replace(' ', '_'),
        decoded.replace('_', ' '),
    ])

    # Optional: Add secure_filename versions too
    base_variants.update([
        secure_filename(v) for v in base_variants
    ])

    return list(base_variants)

@app.route('/api/images/<path:disease_name>/<dataset_type>', methods=['GET']) #List the images
def list_images(disease_name, dataset_type):
    try:
        if dataset_type not in ['train', 'validation', 'all']:
            return jsonify({'error': 'Invalid dataset type'}), 400

        candidate_names = possible_folder_names(disease_name)
        print(f"Trying folder names: {candidate_names}")  # Debug

        images = []
        found_any_folder = False
        search_paths = []

        # Build folder search combinations
        for name_variant in candidate_names:
            if dataset_type == 'all':
                search_paths.extend([
                    (os.path.join(TRAIN_FOLDER, name_variant), 'train'),
                    (os.path.join(VAL_FOLDER, name_variant), 'validation')
                ])
            else:
                base = TRAIN_FOLDER if dataset_type == 'train' else VAL_FOLDER
                search_paths.append((os.path.join(base, name_variant), dataset_type))

        for folder_path, source in search_paths:
            if not os.path.exists(folder_path):
                continue

            found_any_folder = True
            for filename in sorted(os.listdir(folder_path)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        with open(os.path.join(folder_path, filename), 'rb') as f:
                            img_data = f.read()
                            mime_type = 'image/png' if filename.lower().endswith('.png') else 'image/jpeg'
                            images.append({
                                'name': filename,
                                'dataset': source,
                                'data': f'data:{mime_type};base64,{base64.b64encode(img_data).decode("utf-8")}',
                                'mime_type': mime_type
                            })
                    except (IOError, PermissionError):
                        continue

        if not found_any_folder:
            return jsonify({
                'error': 'No folder matched for any variant of "' + disease_name + '"',
                'tried_folders': [p[0] for p in search_paths]
            }), 404

        if not images:
            return jsonify({
                'error': f'Folders found but no images inside.',
                'folders_checked': [p[0] for p in search_paths]
            }), 404

        return jsonify({
            'images': images,
            'original_request': disease_name,
            'folder_variants_checked': candidate_names
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/images/<disease_name>/<dataset_type>/<filename>', methods=['DELETE'])
def delete_image(disease_name, dataset_type, filename):
    """Delete specific image"""
    if dataset_type not in ['train', 'validation']:
        return jsonify({'error': 'Invalid dataset type'}), 400

    folder = os.path.join(TRAIN_FOLDER if dataset_type == 'train' else VAL_FOLDER,
                          secure_filename(disease_name))
    filepath = os.path.join(folder, secure_filename(filename))

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        os.remove(filepath)
        return jsonify({'message': f'Deleted {filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False,port=5000,host='0.0.0.0')
