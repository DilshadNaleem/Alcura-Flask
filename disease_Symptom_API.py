from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from PIL import Image
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import io
import logging
from typing import Tuple, List, Dict, Optional, Union

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DISEASE_INFO_PATH = 'Disease/disease_information.csv'
IMAGE_BASE_PATH = 'Disease/train'
MAX_IMAGES = 4
TOP_N_MATCHES = 2

# Body part detection with synonyms
BODY_PARTS = {
    'skin': {'skin', 'dermis', 'epidermis', 'body', 'all over', 'face', 'arm', 'leg', 'hand', 'foot'},
    'eye': {'eye', 'eyes', 'ocular', 'vision', 'eyelid', 'cornea', 'retina', 'pupil'},
    'mouth': {'mouth', 'oral', 'lips', 'gums', 'tongue', 'tooth', 'teeth', 'palate'},
    'nail': {'nail', 'nails', 'fingernail', 'toenail', 'cuticle'}
}

# Initialize global variables
disease_df = None
vectorizer = None
disease_tfidf = None


def initialize_app():
    """Initialize the application by loading and preprocessing disease data."""
    global disease_df, vectorizer, disease_tfidf

    try:
        # Load disease info
        disease_df = pd.read_csv(DISEASE_INFO_PATH)
        logger.info("Successfully loaded disease information")

        # Preprocess data
        disease_df, vectorizer, disease_tfidf = preprocess_disease_data(disease_df)
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise


def preprocess_disease_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    """Preprocess disease data for dynamic matching."""
    try:
        # Create a combined text field for each disease
        df['combined_text'] = df.apply(lambda row: " ".join([
            str(row['symptoms']),
            str(row['Description']),
            str(row['common_age_group']),
            str(row['first_aid_advice']),
            str(row['cause']),
            str(row['side_effects'])
        ]).lower(), axis=1)

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

        return df, vectorizer, tfidf_matrix
    except Exception as e:
        logger.error(f"Error preprocessing disease data: {str(e)}")
        raise


def get_disease_info(disease_name: str) -> Optional[Dict[str, str]]:
    """Get detailed information about a specific disease."""
    try:
        disease_row = disease_df[disease_df['Disease'].str.lower() == disease_name.lower()].iloc[0]
        return {
            'Disease': disease_row['Disease'],
            'Scientific Name': disease_row['scientific_name'],
            'Description': disease_row['Description'],
            'Symptoms': disease_row['symptoms'],
            'Cause': disease_row['cause'],
            'Side Effects': disease_row['side_effects'],
            'Treatment': disease_row['treatment'],
            'Medications': disease_row['medications'],
            'Prevention': disease_row['prevention'],
            'Severity': disease_row['severity'],
            'Risk Factors': disease_row['risk_factors'],
            'Is Contagious': disease_row['is_contagious'],
            'Common Age Group': disease_row['common_age_group'],
            'Duration': disease_row['duration'],
            'First Aid Advice': disease_row['first_aid_advice'],
            'Source': disease_row['Source_of_information']
        }
    except IndexError:
        logger.warning(f"Disease not found: {disease_name}")
        return None
    except Exception as e:
        logger.error(f"Error getting disease info for {disease_name}: {str(e)}")
        return None


def get_sample_images(disease_name: str, base_path: str = IMAGE_BASE_PATH, max_images: int = MAX_IMAGES) -> List[str]:
    """Get sample images for a disease as base64 encoded strings."""
    try:
        image_folder = os.path.join(base_path, disease_name)
        if not os.path.exists(image_folder):
            logger.info(f"No image folder found for disease: {disease_name}")
            return []

        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files = image_files[:max_images]

        encoded_images = []
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            try:
                with Image.open(img_path) as img:
                    # Convert image to RGB if it's not
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    encoded_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")

        return encoded_images
    except Exception as e:
        logger.error(f"Error getting images for {disease_name}: {str(e)}")
        return []


def analyze_input_and_match_diseases(user_input: str, top_n: int = TOP_N_MATCHES) -> Tuple[List[str], Dict[str, int]]:
    """Analyze user input and find matching diseases."""
    try:
        input_text = user_input.lower()

        # Detect affected body parts
        affected_parts = set()
        for part, synonyms in BODY_PARTS.items():
            if any(synonym in input_text for synonym in synonyms):
                affected_parts.add(part)

        # Vectorize user input
        input_tfidf = vectorizer.transform([input_text])

        # Calculate cosine similarity between input and diseases
        similarities = cosine_similarity(input_tfidf, disease_tfidf).flatten()

        # Replace NaN values with 0 and filter out negative scores
        similarities = np.nan_to_num(similarities, nan=0.0)
        similarities = np.clip(similarities, 0, None)

        # Get disease names and similarity scores
        disease_names = disease_df['Disease'].values
        scored_diseases = list(zip(disease_names, similarities))

        # Filter by body part if any were detected
        if affected_parts:
            filtered_diseases = []
            for disease, score in scored_diseases:
                disease_row = disease_df[disease_df['Disease'] == disease].iloc[0]
                disease_text = disease_row['combined_text'].lower()

                # Check if disease mentions any of the affected body parts
                matches_body_part = False
                for part in affected_parts:
                    if any(synonym in disease_text for synonym in BODY_PARTS[part]):
                        matches_body_part = True
                        break

                if matches_body_part:
                    filtered_diseases.append((disease, score))

            if filtered_diseases:
                scored_diseases = filtered_diseases

        # Sort by similarity score
        scored_diseases.sort(key=lambda x: x[1], reverse=True)

        # Filter out diseases with zero similarity
        scored_diseases = [(disease, score) for disease, score in scored_diseases if score > 0]

        if not scored_diseases:
            return [], {}

        # Calculate confidence percentages
        max_score = max(score for _, score in scored_diseases)
        confidences = {
            disease: min(100, int((score / max_score) * 100))
            for disease, score in scored_diseases
        }

        # Get top matches
        best_matches = [disease for disease, _ in scored_diseases[:top_n]]

        return best_matches, confidences
    except Exception as e:
        logger.error(f"Error analyzing input: {str(e)}")
        return [], {}


@app.route('/analyze', methods=['POST'])
def analyze_symptoms():
    """Endpoint to analyze symptoms and return matching diseases."""
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Please provide symptoms description',
                'matches': []
            }), 400

        user_input = data['symptoms'].strip()
        if not user_input:
            return jsonify({
                'status': 'error',
                'message': 'Please enter some symptoms to analyze',
                'matches': []
            }), 400

        matched_diseases, confidences = analyze_input_and_match_diseases(user_input)

        results = []
        for disease in matched_diseases:
            confidence = confidences.get(disease, 0)
            disease_info = get_disease_info(disease)
            images = get_sample_images(disease)

            if disease_info:
                results.append({
                    'disease': disease,
                    'confidence': confidence,
                    'info': disease_info,
                    'images': images
                })

        if not results:
            return jsonify({
                'status': 'success',
                'message': 'No matching disease found from the description',
                'matches': []
            }), 200

        return jsonify({
            'status': 'success',
            'message': f'Found {len(results)} potential matches',
            'matches': results
        }), 200

    except Exception as e:
        logger.error(f"Error in analyze_symptoms: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An internal server error occurred',
            'matches': []
        }), 500


@app.route('/diseases', methods=['GET'])
def list_diseases():
    """Endpoint to list all available diseases."""
    try:
        diseases = disease_df['Disease'].tolist()
        return jsonify({
            'status': 'success',
            'message': f'Found {len(diseases)} diseases',
            'diseases': diseases
        }), 200
    except Exception as e:
        logger.error(f"Error in list_diseases: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An internal server error occurred',
            'diseases': []
        }), 500


@app.route('/disease/<name>', methods=['GET'])
def get_disease(name):
    """Endpoint to get details about a specific disease."""
    try:
        disease_info = get_disease_info(name)
        if not disease_info:
            return jsonify({
                'status': 'error',
                'message': 'Disease not found',
                'disease': None,
                'info': {},
                'images': []
            }), 404

        images = get_sample_images(name)
        return jsonify({
            'status': 'success',
            'message': 'Disease information retrieved',
            'disease': name,
            'info': disease_info,
            'images': images
        }), 200
    except Exception as e:
        logger.error(f"Error in get_disease: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'An internal server error occurred',
            'disease': None,
            'info': {},
            'images': []
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Service is operational',
        'services': {
            'disease_detection': True,
            'image_processing': True,
            'symptom_analysis': True
        }
    }), 200


if __name__ == '__main__':
    try:
        initialize_app()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")