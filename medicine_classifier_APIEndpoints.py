import base64
import json
import os
from flask import Flask, request, jsonify, send_file
from sympy.plotting.textplot import rescale
from torch.fx.experimental.unification.multipledispatch.dispatcher import source
from torchvision.transforms.v2.functional import horizontal_flip
from urllib.parse import unquote
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


matplotlib.use('Agg')

app = Flask(__name__)

Upload_Folder = 'uploads'
Train_folder = 'Pil-DataSet/train'
Validation_folder = 'Pil-DataSet/validation'
Pill_INFO_FILE = "Pil-DataSet/drug_information.csv"
Model_File = "medicine_classifier_model.h5"

os.makedirs(Upload_Folder, exist_ok=True)
os.makedirs(Train_folder, exist_ok=True)
os.makedirs(Validation_folder, exist_ok=True)

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 4
CONFIDENCE_THRESHOLD = 0.7

medicine_df = pd.DataFrame(columns=[
    'class_name', 'dosage', 'use', 'price', 'side_effects', 'dosage_form', 'Scientific_Name',
    'max_dose', 'administration', 'indications', 'precautions', 'serious_effects',
    'contraindications', 'Source_of_information'
])

if os.path.exists(Pill_INFO_FILE):
    medicine_df = pd.read_csv(Pill_INFO_FILE)

model = None
if os.path.exists(Model_File):
    try:
        model = tf.keras.models.load_model(Model_File)
        print("Pre-trained model loaded")
    except:
        print("Couldn't load a pre-trained model")

def build_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
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

@app.route("/medicine/getAllMedicine", methods=['GET'])
def get_pills():
    return jsonify({"medicine": medicine_df['class_name'].unique().tolist()})

@app.route("/medicine/getAllMedicine&Description", methods=["GET"])
def get_all_pills():
    all_pills = medicine_df.to_dict('records')
    return jsonify(all_pills)


@app.route("/api/AddPill", methods=['POST'])
def add_pills():
    try:
        medicine_data = request.form.get('medicineData')
        if not medicine_data:
            return jsonify({'error': 'Pill Data is required'}), 400

        medicine_info = json.loads(medicine_data)

        if 'class_name' not in medicine_info:
            return jsonify({'error': 'Pill name is required'}), 400

        medicine_name = medicine_info['class_name']
        folder_name = ' '.join(medicine_name.split())

        existing_index = medicine_df.index[medicine_df['class_name'] == medicine_name].tolist()

        new_row = {
            'class_name': '', 'dosage': '', 'use': '', 'price': '', 'side_effects': '', 'dosage_form': '',
            'Scientific_Name': '', 'max_dose': '', 'administration': '', 'indications': '', 'precautions': '',
            'serious_effects': '', 'contraindications': '', 'Source_of_information': ''
        }

        new_row.update({k: v for k, v in medicine_info.items() if k in new_row})

        if existing_index:
            medicine_df.loc[existing_index[0]] = new_row
        else:
            medicine_df.loc[len(medicine_df)] = new_row

        medicine_df.to_csv(Pill_INFO_FILE, index=False)

        train_folder = os.path.join(Train_folder, folder_name)
        val_folder = os.path.join(Validation_folder, folder_name)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        train_count = 0
        if 'trainImages' in request.files:
            for file in request.files.getlist('trainImages'):
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    try:
                        file.save(os.path.join(train_folder, filename))
                        train_count += 1
                    except Exception as e:
                        app.logger.error(f"Error saving training image {filename}: {str(e)}")
                        continue

        val_count = 0
        if 'valImages' in request.files:
            for file in request.files.getlist('valImages'):
                if file.filename != '':
                    filename = secure_filename(file.filename)
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
            'message': f'Pill information saved with {train_count} training images and {val_count} validation images',
            'folder_name': folder_name
        }), 200

    except Exception as e:
        app.logger.error(f"Error in add_pill: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route("/api/deletePill/<medicine_name>", methods=['DELETE']) #Delete Medicine
def delete_pill(medicine_name):

    train_folder = os.path.join(Train_folder, medicine_name)
    val_folder = os.path.join(Validation_folder, medicine_name)

    try:
        if os.path.exists(train_folder):
            shutil.rmtree(train_folder)
        if os.path.exists(val_folder):
            shutil.rmtree(val_folder)

        global medicine_df
        original_count = len(medicine_df)

        medicine_df = medicine_df[~medicine_df['class_name'].str.strip().str.lower(). eq(medicine_name.strip().lower())]

        new_count = len(medicine_df)

        if original_count == new_count:
            return jsonify({'warning': f'Medicine "{medicine_name}" not found in dataframe'}), 404

        medicine_df.to_csv(Pill_INFO_FILE, index=False)

        return jsonify({'message' : f'Deleted all images and information for {medicine_name}'}), 200
    except Exception as e:
        return jsonify({'error':f'Could not Delete: {str(e)}'}), 500


@app.route('/api/Medicinetrain', methods=['POST'])
def train_model():
    try:
        tf.config.run_functions_eagerly(True)

        train_classes = [d for d in os.listdir(Train_folder) if os.path.isdir(os.path.join(Train_folder,d))]
        val_classes = [d for d in os.listdir(Validation_folder) if os.path.isdir(os.path.join(Train_folder, d))]

        if not train_classes:
            return jsonify({'error': 'No Validation data found! Please add images first'}), 400

        class CustomImageDataGenerator(ImageDataGenerator):
            def _init_(self, **kwargs):
                super()._init_(**kwargs)

        train_datagen = CustomImageDataGenerator(
            rescale = 1.0 / 255,
            rotation_range=20,
            zoom_range = 0.2,
            horizontal_flip= True,
            fill_mode = 'nearest'
        )

        val_datagen = CustomImageDataGenerator(rescale = 1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            Train_folder,
            target_size = (IMG_HEIGHT, IMG_WIDTH),
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            shuffle = True,
        )

        val_generator = val_datagen.flow_from_directory(
            Validation_folder,
            target_size = (IMG_HEIGHT,IMG_WIDTH),
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            shuffle = False,
        )

        global model
        needs_build = (
            model is None or
            model.output_shape[1] != train_generator.num_classes or
            not hasattr(model, 'optimizer') or
            not isinstance(model.optimizer, tf.keras.optimizers.Optimizer)
        )

        if needs_build:
            model = build_model(train_generator.num_classes)
            print("Built new model With " , train_generator.num_classes, "classes")
        else:
            model.compile (
                optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            print("Recompiled existing model")

        epochs = int(request.form.get('epochs', 10))
        history = model.fit(
            train_generator,
            validation_data = val_generator,
            epochs = epochs,
            steps_per_epoch = max(1, len(train_generator)),
            validation_steps = max(1, len(val_generator)),
            verbose = 1
        ).history

        model.save(Model_File)
        print("Model Saved successfully")

        history = {k: [float(v) for v in vals] for k, vals in history.items()}

        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.plot(history['accuracy'], label = 'Training Accuracy')
        plt.plot(history['val_accuracy'], label = 'Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label = 'Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        chart_bytes = io.BytesIO()
        plt.savefig(chart_bytes, format='png')
        plt.close()
        chart_bytes.seek(0)
        chart_base64 = base64.b64encode(chart_bytes.read()). decode('utf-8')

        return jsonify({
            'message': 'Model training Completed Successfully!',
            'history': history,
            'chart': f"data:image/png;base64,{chart_base64}"
        })

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return jsonify({
            'error' : f'Training failed: {str(e)}',
            'traceback' : traceback.format_exc()
        }), 500


@app.route('/api/Medicine/Predict', methods=['POST'])  # Predict Medicine
def medicinePredict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    if model is None:
        return jsonify({'error': 'No trained model found. Please train a model first'})

    file = request.files['image']

    try:
        img = Image.open(file.stream).resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img) / 255.0

        # Check if image is grayscale (2D) and convert to RGB (3 channels)
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        # Ensure image has 3 channels even if it's already RGB
        img_array = img_array[..., :3]

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx])

        train_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
            Train_folder,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
        )

        class_names = {v: k for k, v in train_generator.class_indices.items()}

        # Initialize response with default values for unrecognized medicine
        response = {
            'predicted_class': 'unrecognized medicine',
            'confidence': confidence,
            'is_recognized': False,
            'medicine_info': None
        }

        # Only consider it recognized if confidence meets threshold AND class exists in our dataset
        if confidence >= CONFIDENCE_THRESHOLD and class_idx in class_names:
            predicted_class = class_names[class_idx]
            response['predicted_class'] = predicted_class
            response['is_recognized'] = True

            medicine_info = medicine_df[medicine_df['class_name'] == predicted_class]
            if not medicine_info.empty:
                response['medicine_info'] = medicine_info.iloc[0].to_dict()

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Could not process image: {str(e)}'}), 500


@app.route('/api/Medicine/Validation_results', methods=['GET']) #Get Validations
def get_validation_result():

    if model is None:
        return jsonify({"error": "No trained model found"})

    val_datagen = ImageDataGenerator(rescale= 1.0 / 255)
    val_generator = val_datagen.flow_from_directory(
        Validation_folder,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    class_names = {v: k for k, v in val_generator.class_indices.items()}
    restults = []

    for i in range (min(20, len(val_generator))):
        img, true_label = val_generator[i]
        true_class_idx = np.argmax(true_label[0])
        true_class = class_names[true_class_idx]

        prediction = model.predict(img)
        pred_class_idx = np.argmax(prediction[0])
        pred_class = class_names[pred_class_idx]
        confidence = float(prediction[0][pred_class_idx])

        img_bytes = io.BytesIO()
        plt.imshow(img[0])
        plt.axis('off')
        plt.savefig(img_bytes, format='png', bbox_inches = 'tight', pad_inches=0)
        plt.close()
        img_bytes.seek(0)

        restults.append({
            'true_class': true_class,
            'predicted_class' : pred_class,
            'confidence' : confidence,
            'is_correct': true_class == pred_class,
            'image' : f"data:image/png;base64,{base64.b64encode(img_bytes.read()).decode('utf-8')}"
        })

    return jsonify(restults)

@app.route('/api/Medicine/Validation_results/search', methods=['GET'])
def search_validation():

    if model is None:
        return jsonify({'error': 'No train model found'}), 500

    class_names = request.args.get('class','').lower()
    is_correct = request.args.get('is_correct', None)

    all_results = get_validation_result().get_json()

    filtered_results = []
    for result in all_results:
        if class_names and (class_names not in result['true_class'].lower() and
        class_names not in result['predicted_class'].lower()):
            continue

        if is_correct is not None:
                expected_correct = is_correct.lower() == 'true'
                if result['is_correct'] != expected_correct:
                    continue

        filtered_results.append(result)

    return jsonify(filtered_results)


@app.route('/api/Medicine/images/<medicine_name>', methods=['POST']) #Add images to folder
def upload_images (medicine_name):
    print(f"Received upload for: {medicine_name}")

    if 'images' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    dataset_type = request.form.get('data_type', 'train')
    if dataset_type not in ['train', 'validation']:
        return jsonify({'error' : 'Invalid dataset type, Use "train" or "validation"'}), 400


    target_folder = Train_folder if dataset_type == 'train' else Validation_folder
    medicine_folder = os.path.join(target_folder, medicine_name)
    os.makedirs(medicine_folder, exist_ok=True)

    files = request.files.getlist('images')
    saved_files = []

    for file in files:
        try:
            img = Image.open(file.stream)
            img.verify()

            filename = secure_filename(file.filename)
            filepath = os.path.join(medicine_folder, filename)
            file.seek(0)
            file.save(filepath)
            saved_files.append(filename)
        except Exception as e:
            continue

    return jsonify({
        'message' : f'Uploaded {len(saved_files)} images for {medicine_name}',
        'saved_files' : saved_files
    })

@app.route('/api/edit/Medicine/<old_name>', methods=['POST'])  #Edit
def edit_medicine(old_name):

    try:
        data = request.json
        new_name = data.get('new_name')
        print(medicine_df['class_name'].values)
        if not new_name:
            return jsonify({'error': 'new name is required'}), 400

        if old_name not in medicine_df['class_name'].values:
            return jsonify({'error': f'Medicine "{old_name}" not found' }), 400

        idx = medicine_df[medicine_df['class_name'] == old_name].index[0]

        for column in medicine_df.columns:
            if column in data and column != 'class_name':
                medicine_df.at[idx, column] = data[column]

        medicine_df.at[idx, 'class_name'] = new_name

        medicine_df.to_csv(Pill_INFO_FILE, index=False)

        if old_name != new_name:
            old_train = os.path.join(Train_folder, old_name)
            new_train = os.path.join(Train_folder, new_name)
            old_val = os.path.join(Validation_folder, old_name)
            new_val = os.path.join(Validation_folder, new_name)

            if os.path.exists(old_train):
                os.rename(old_train, new_train)

            if os.path.exists(old_val):
                os.rename(old_val, new_val)

        return jsonify({
            'message' : f'Updated sieases "{old_name}"',
            'old_folder_name': old_name if old_name != new_name else None,
            'new_folder_name' : new_name if old_name != new_name else None,
            'updated_fields': list(data.keys())
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/medicine/SearchByName/<medicine_name>', methods=['GET']) #Search by name view all details
def get_medicine_info(medicine_name):
    try:
        current_df = pd.read_csv(Pill_INFO_FILE)
        medicine_data = current_df[current_df['class_name'].str.strip().str.lower() == medicine_name.strip().lower()]

        if medicine_data.empty:
            return jsonify({'error': f'Medicine "{medicine_name}" not found'}), 404

        result = medicine_data.iloc[0].to_dict()
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Server Error: "{str(e)}"'}), 500

def possible_folder_names(name):
    decoded = unquote(name.replace('+', ' '))
    base_varients = set([
    decoded,
    decoded.replace(' ', '_'),
    decoded.replace('_', ' ')
])
    base_varients.update([
    secure_filename(v) for v in base_varients
    ])

    return list(base_varients)

@app.route('/api/images/<path:medicine_name>/<dataset_type>', methods=['GET']) # View all images based on train vali and all
def list_images(medicine_name, dataset_type):
    try:
        if dataset_type not in ['train', 'validation', 'all']:
            return jsonify({'error': 'Invalid dataset type'}), 400

        candidate_names = possible_folder_names(medicine_name)
        print(f"Trying folder names: {candidate_names}")

        images = []
        found_any_folder = False
        search_paths = []

        for name_variant in candidate_names:
            if dataset_type == 'all':
                search_paths.extend({
                    (os.path.join(Train_folder, name_variant), 'train'),
                    (os.path.join(Validation_folder, name_variant), 'validation')
                })
            else:
                base = Train_folder if dataset_type == 'train' else Validation_folder
                search_paths.append((os.path.join(base, name_variant), dataset_type))

        for folder_path, source in search_paths:
            if not os.path.exists(folder_path):
                continue

            found_any_folder = True
            for filename in sorted(os.listdir(folder_path)):
                    if filename.lower().endswith(('.png','jpg', 'jpeg')):

                        try:
                            with open(os.path.join(folder_path, filename), 'rb') as f:
                                img_data = f.read()
                                mime_type = 'image/pmg' if filename.lower().endswith('png') else 'image/jpeg'
                                images.append({
                                    'name' : filename,
                                    'dataset' : source,
                                    'data' : f'data: {mime_type}; base64, {base64.b64encode(img_data).decode("utf-8")}',
                                    'mime_type' : mime_type
                                })
                        except (IOError, PermissionError):
                            continue

        if not found_any_folder:
            return jsonify({
                'error': 'Now folder matched for any variant of "' + medicine_name + '"',
                'tried_folders' : [p[0] for p in search_paths]
            }), 404

        if not images:
            return jsonify({
                'error' : f'Folders found but no image inside.',
                'folders_checked' : [p[0] for p in search_paths]
            }), 404

        return jsonify({
            'images' : images,
            'original_request' : medicine_name,
            'folder_variants_checked' : candidate_names
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/images/<medicine_name>/<dataset_type>/<filename>', methods= ['DELETE']) # Delete images
def delete_images (medicine_name, dataset_type, filename):

    if dataset_type not in ['train', 'validation']:
        return jsonify({'error' : 'Invalid dataset type'}), 400

    folder = os.path.join(Train_folder if dataset_type == 'train' else Validation_folder,
                          secure_filename(medicine_name))

    filepath = os.path.join(folder, secure_filename(filename))

    if not os.path.exists(filepath):
        return ({'error' : 'File not Found'}),404

    try:
        os.remove(filepath)
        return jsonify({'message' : f'Deleted {filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')
