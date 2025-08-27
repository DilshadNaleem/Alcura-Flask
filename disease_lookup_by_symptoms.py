import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load disease info
disease_df = pd.read_csv('Disease/disease_information.csv')


# Preprocess disease data for dynamic matching
def preprocess_disease_data(df):
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


disease_df, vectorizer, disease_tfidf = preprocess_disease_data(disease_df)

# Body part detection with synonyms
body_parts = {
    'skin': {'skin', 'dermis', 'epidermis', 'body', 'all over', 'face', 'arm', 'leg', 'hand', 'foot'},
    'eye': {'eye', 'eyes', 'ocular', 'vision', 'eyelid', 'cornea', 'retina', 'pupil'},
    'mouth': {'mouth', 'oral', 'lips', 'gums', 'tongue', 'tooth', 'teeth', 'palate'},
    'nail': {'nail', 'nails', 'fingernail', 'toenail', 'cuticle'}
}


# Helper: Show info about the disease
def show_disease_info(disease_name):
    try:
        disease_row = disease_df[disease_df['Disease'].str.lower() == disease_name.lower()].iloc[0]

        print(f"\n{'=' * 40}")
        print(f"Disease: {disease_row['Disease']}")
        print(f"Scientific Name: {disease_row['scientific_name']}")
        print(f"Description: {disease_row['Description']}")
        print(f"Symptoms: {disease_row['symptoms']}")
        print(f"Cause: {disease_row['cause']}")
        print(f"Side Effects: {disease_row['side_effects']}")
        print(f"Treatment: {disease_row['treatment']}")
        print(f"Medications: {disease_row['medications']}")
        print(f"Prevention: {disease_row['prevention']}")
        print(f"Severity: {disease_row['severity']}")
        print(f"Risk Factors: {disease_row['risk_factors']}")
        print(f"Is Contagious: {disease_row['is_contagious']}")
        print(f"Common Age Group: {disease_row['common_age_group']}")
        print(f"Duration: {disease_row['duration']}")
        print(f"First Aid Advice: {disease_row['first_aid_advice']}")
        print(f"Source: {disease_row['Source_of_information']}")
        print(f"{'=' * 40}\n")

    except IndexError:
        print(f"No detailed info found for disease: {disease_name}")


# Helper: Show images from Disease/train/<disease>
def show_sample_images(disease_name, base_path='Disease/train', max_images=4):
    image_folder = os.path.join(base_path, disease_name)
    if not os.path.exists(image_folder):
        print(f"No images found for {disease_name} in {image_folder}")
        return

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files = image_files[:max_images]

    if not image_files:
        print(f"No image files found for {disease_name}")
        return

    plt.figure(figsize=(12, 3))
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_folder, img_file)
        try:
            img = Image.open(img_path)
            plt.subplot(1, len(image_files), idx + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{disease_name}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    plt.tight_layout()
    plt.show()


def analyze_input_and_match_diseases(user_input, top_n=2):
    input_text = user_input.lower()

    # Detect affected body parts
    affected_parts = set()
    for part, synonyms in body_parts.items():
        if any(synonym in input_text for synonym in synonyms):
            affected_parts.add(part)

    # Vectorize user input
    input_tfidf = vectorizer.transform([input_text])

    # Calculate cosine similarity between input and diseases
    similarities = cosine_similarity(input_tfidf, disease_tfidf).flatten()

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
                if any(synonym in disease_text for synonym in body_parts[part]):
                    matches_body_part = True
                    break

            if matches_body_part:
                filtered_diseases.append((disease, score))

        if filtered_diseases:
            scored_diseases = filtered_diseases

    # Sort by similarity score
    scored_diseases.sort(key=lambda x: x[1], reverse=True)

    # Calculate confidence percentages (normalize to 0-100 scale)
    max_score = scored_diseases[0][1] if scored_diseases else 1
    confidences = {disease: int((score / max_score) * 100) for disease, score in scored_diseases}

    # Get top matches
    best_matches = [disease for disease, _ in scored_diseases[:top_n]]

    # Debug output
    print("\nMatching Details:")
    for disease, confidence in list(confidences.items())[:5]:
        print(f"{disease}: {confidence}% confidence")

    return best_matches, confidences


# === MAIN ===
if __name__ == '__main__':
    print("Describe your symptoms (e.g. 'itching all the time, round shapes, on the skin'):")
    user_input = input("Your description: ").strip()

    if not user_input:
        print("Please enter some symptoms to analyze.")
    else:
        matched_diseases, confidences = analyze_input_and_match_diseases(user_input)

        if not matched_diseases:
            print("\nNo matching disease found from the description.")
        else:
            print(f"\nTop Matches Based on Description:")
            for disease in matched_diseases:
                confidence = confidences.get(disease, 0)
                print(f"\n{'=' * 40}")
                print(f"MATCH CONFIDENCE: {confidence}%")
                print(f"{'=' * 40}")
                show_disease_info(disease)
                show_sample_images(disease)

    print("\nDone.")