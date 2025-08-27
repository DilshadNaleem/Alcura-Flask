import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from PIL import Image,UnidentifiedImageError
import shutil
from sklearn.model_selection import train_test_split

# Dataset paths
train_path = "Pil-DataSet/train"
val_path = "Pil-DataSet/validation"
unknown_path = "Pil-DataSet/unknown"  # New directory for non-medicine images

# Create unknown class directory if it doesn't exist
os.makedirs(os.path.join(train_path, "unknown"), exist_ok=True)
os.makedirs(os.path.join(val_path, "unknown"), exist_ok=True)

# Add some non-medicine images to the unknown class (you should add your own images here)
# This is just a placeholder - in practice, you should collect many diverse non-medicine images
sample_unknown_images = [
    "susantika.jpeg",
    "paneerbaratha.jpg",

]

for img_path in sample_unknown_images:
    if os.path.exists(img_path):
        shutil.copy(img_path, os.path.join(train_path, "unknown", os.path.basename(img_path)))

# Image parameters
img_height, img_width = 224, 224
batch_size = 4
confidence_threshold = 0.7  # Minimum confidence to accept prediction

# Create data generators with augmented unknown class
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',

)

# Full training generator
full_train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,

)

# Validation generator
val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,

)

def create_drug_info_dataframe():
    data = [
        {
            'class_name': 'Brufan',
            'dosage': '600mg',
            'use': 'Reduces pain, inflammation, and fever.',
            'price': 'Approximately Rs.4 per pill.',
            'side_effects': 'Stomach irritation, nausea, dizziness, headache.',
            'dosage_form' : 'Tablets, oral suspension.',
            'Scientific_Name' : 'Ibuprofen',
            'max_dose' : 'Adults: 1200 mg/day; Children: 40 mg/kg/day.',
            'administration': 'Oral intake; may be taken with or without food.',
            'indications': 'Pain relief (headache, dental pain, muscle pain), anti-inflammatory for conditions like arthritis, fever reduction.',
            'precautions': 'Use the lowest effective dose for the shortest duration; monitor for gastrointestinal issues.',
            'serious_effects': 'Gastrointestinal bleeding, kidney damage, liver dysfunction.',
            'contraindications': 'Hypersensitivity to ibuprofen or other NSAIDs, active gastrointestinal bleeding, severe heart failure, liver or kidney impairment.',
            'Source_of_information': 'RxList,MIMS, Medicines.org.uk'
        },
        {
            'class_name': 'Paracetamol',
            'dosage': '500mg per tablet ',
            'use': 'Pain and fever relief',
            'price': 'Approximately Rs.2 per pill.',
            'side_effects': 'Rare: May Include allergic reactions.',
            'dosage_form': 'Tablets, oral suspension, suppositories.',
            'Scientific_Name': 'also known as Acetaminophen in the US and Canada',
            'max_dose': 'Adults: 4000 mg/day; Children: 75 mg/kg/day.',
            'administration': 'Oral intake; may be taken with or without food.',
            'indications': 'Mild to moderate pain relief (headache, menstrual pain, toothache), fever reduction.',
            'precautions': 'Avoid exceeding recommended dose to prevent liver damage.',
            'serious_effects': ' Liver toxicity, especially with overdose or chronic use.',
            'contraindications': 'Severe liver disease, hypersensitivity to paracetamol.',
            'Source_of_information':''
        },
        {
            'class_name': 'Aspirin',
            'dosage': '325 mg',
            'use': 'Pain relief, anti-inflammatory, heart attack and stroke prevention',
            'price': 'Rs.3 per pill',
            'side_effects': 'Gastric irritation, gastrointestinal bleeding, allergic reactions.',
            'dosage_form': ' Tablets, chewable tablets, enteric-coated tablets, suppositories.',
            'Scientific_Name': 'Acetylsalicylic Acid',
            'max_dose': 'Adults: 4000 mg/day; for cardiovascular prevention: 75-81 mg/day.',
            'administration': 'Oral intake; take with a full glass of water.',
            'indications': ' Pain relief (headache, muscle pain), anti-inflammatory for conditions like arthritis, cardiovascular protection (heart attack, stroke).',
            'precautions': 'Use with caution in individuals with gastrointestinal issues or bleeding disorders.',
            'serious_effects': 'Gastrointestinal bleeding, Reye s syndrome in children with viral infections.',
            'contraindications': 'Hypersensitivity to aspirin, history of gastrointestinal bleeding, children with viral infections',
            'Source_of_information': 'RxList, MedicineNet,Drugs.com'
        },
        {
            'class_name': 'unknown',
            'dosage': 'Not a medicine',
            'use': 'Not a medicine',
            'price': 'Not a medicine',
            'side_effects': 'Not a medicine',
            'dosage_form': 'Not a medicine',
            'Scientific_Name': 'Not a medicine',
            'max_dose': 'Not a medicine',
            'administration': 'Not a medicine',
            'indications': 'Not a medicine',
            'precautions': 'Not a medicine',
            'serious_effects': 'Not a medicine',
            'contraindications': 'Not a medicine',
            'Source_of_information': 'Not a medicine'
        },
        {
            'class_name': 'Diclofenac',
            'dosage': '50-75 mg orally 2-3 times daily (maximum 150 mg per day)',
            'use': 'Pain relief, inflammation reduction in conditions like arthirits, muscle pain, menstrual cramps, gout, migrane',
            'price': 'Rs.4 per pill',
            'side_effects': 'Nausea, heartburn, dizziness, stomach upset, headache, skin rash.',
            'dosage_form': ' Tablets, capsules, injections, creams, suppositories.',
            'Scientific_Name': 'Diclofenac Sodium or Diclofenac Potassium',
            'max_dose': 'Usually 150 mg/day orally, 75 mg/day intramuscularly.',
            'administration': 'Oral intake; take with a full glass of water.',
            'indications': 'Rheumatoid arthritis, osteoarthritis, ankylosing spondylitis, acute pain, dysmenorrhea, post-operative pain.',
            'precautions': '	Use cautiously in hypertension, heart disease, GI ulcers, asthma, renal impairment.',
            'serious_effects': '	Stomach bleeding, liver damage, kidney problems, heart attack, stroke, anaphylaxis.',
            'contraindications': 'Known allergy to NSAIDs, history of asthma induced by NSAIDs, active peptic ulcer, severe heart failure, late pregnancy (3rd trimester)',
            'Source_of_information': 'Drugs.com, MedinePlus, BNF'
        },

        {
            'class_name': 'Amoxicillin',
            'dosage': 'Adults: 250–500 mg every 8 hours or 500–875 mg every 12 hours.'
                      'Children: 20–45 mg/kg/day in divided doses.',
            'use' : 'Treatment of bacterial infections: respiratory tract infections, urinary tract infections, skin infections, otitis media, sinusitis, dental abscess',
            'price' : 'Approx. Rs. 5–12 per 500 mg capsule/tablet, depending on brand',
            'side_effects': 'Nausea, diarrhea, skin rash, vomiting, headache, changes in taste',
            'dosage_form' : '	Tablets, capsules, oral suspension, chewable tablets, injections',
            'Scientific_Name' : 'Amoxicillin (a semi-synthetic penicillin)',
            'max_dose' : 'Up to 4 g/day for adults in severe infections',
            'administration' : 'Oral (preferably before meals), or intravenous in hospital settings',
            'indications' : 'Bacterial infections including pneumonia, bronchitis, tonsillitis, UTIs, ENT infections, Helicobacter pylori eradication',
            'precautions' : 'Use with caution in renal impairment, history of allergy to penicillin, monitor for superinfections',
            'serious_effects' : 'Anaphylaxis, Clostridium difficile-associated diarrhea, liver dysfunction, blood disorders',
            'contraindications' : 'Known hypersensitivity to penicillins or beta-lactam antibiotics, history of severe allergic reaction',
            'Source_of_information' : 'Drugs.com, MedinePlus, BNF'
        },

        {
            'class_name' : 'Azithromycin',
            'dosage': '	Adults: 500 mg on day 1, followed by 250 mg once daily on days 2–5. '
                      'Or 500 mg once daily for 3 days depending on infection.',
            'use': 'Treats respiratory infections (bronchitis, pneumonia), sinusitis, skin infections, STDs, ear infections, throat infections',
            'price': 'Around Rs. 40–90 per 500 mg tablet, depending on brand',
            'side_effects': 'Nausea, diarrhea, stomach pain, vomiting, headache, dizziness',
            'dosage_form': 'Tablets, oral suspension, powder for suspension, injections',
            'Scientific_Name': 'Azithromycin Dihydrate (common form)',
            'max_dose': 'Typically 500 mg/day, up to 2 g single dose for certain STDs (under supervision)',
            'administration': 'Orally, 1 hour before or 2 hours after meals; IV in hospital settings',
            'indications': 'Community-acquired pneumonia, pharyngitis, tonsillitis, skin infections, chlamydia, gonorrhea, sinusitis, otitis media',
            'precautions': 'Use cautiously in liver disease, prolonged QT interval, kidney disease, myasthenia gravis',
            'serious_effects': 'QT prolongation, hepatotoxicity, Clostridium difficile-associated diarrhea, allergic reactions',
            'contraindications': 'Hypersensitivity to macrolides, history of jaundice or liver dysfunction with prior azithromycin use',
            'Source_of_information': 'Drugs.com, PubChem, BNF'
        },

        {
            'class_name' : 'Doxycycline',
            'dosage': '	Adults: 100 mg twice daily on day 1, then 100 mg once daily (or 50 mg twice daily).'
                      'Severe infections: Continue 100 mg twice daily.',
            'use': 'Treats bacterial infections, acne, malaria prophylaxis, chlamydia, gonorrhea, periodontitis, respiratory infections, Lyme disease',
            'price': 'Approx. Rs. 10–25 per 100 mg capsule/tablet, depending on the brand',
            'side_effects': 'Nausea, vomiting, diarrhea, photosensitivity, esophagitis, abdominal pain',
            'dosage_form': 'Tablets, capsules, oral suspension, injection, delayed-release tablets',
            'Scientific_Name': 'Doxycycline hyclate or Doxycycline monohydrate',
            'max_dose': 'Generally 200 mg/day, but may vary based on indication',
            'administration': 'Oral (with a full glass of water, sitting upright), preferably with food to reduce GI upset; IV for hospital use',
            'indications': 'Acne vulgaris, sexually transmitted infections, anthrax, rickettsial infections, cholera, malaria prevention, respiratory tract infections',
            'precautions': 'Use caution in liver impairment, photosensitivity, avoid in pregnancy and children under 8 unless essential',
            'serious_effects': 'Hepatotoxicity, intracranial hypertension, esophageal ulceration, severe skin reactions, superinfections',
            'contraindications': 'Hypersensitivity to tetracyclines, pregnancy, children <8 years (due to permanent teeth discoloration and bone growth inhibition)',
            'Source_of_information': 'Drugs.com, PubChem, BNF'
        },

        {
            'class_name': 'Losartan',
            'dosage': 'Adults: Starting dose usually 50 mg once daily; maintenance dose 25–100 mg/day in 1 or 2 divided doses.'
                      'For elderly or liver impairment: Start with 25 mg.',
            'use': 'Treatment of hypertension (high blood pressure), heart failure, diabetic nephropathy, and to reduce the risk of stroke in patients with hypertension',
            'price': 'Approx. Rs. 15–40 per 50 mg tablet, depending on the brand (e.g., Cozaar, Losan, Losartas)',
            'side_effects': 'Dizziness, fatigue, upper respiratory infection, low blood pressure, back pain, increased potassium levels',
            'dosage_form': 'Tablets (25 mg, 50 mg, 100 mg); sometimes available in combination with hydrochlorothiazide',
            'Scientific_Name': 'Losartan potassium',
            'max_dose': '100 mg/day (can be given as 50 mg twice daily or 100 mg once daily)',
            'administration': 'Oral, with or without food; taken once or twice daily',
            'indications': '- Essential hypertension'
                           '- Heart failure (with ACE intolerance)'
                           '- Chronic kidney disease in diabetes'
                           '- Stroke risk reduction',
            'precautions': '- Monitor in renal/hepatic impairment, electrolyte imbalance, volume depletion'
                           '- Avoid potassium supplements unless prescribed',
            'serious_effects': 'Hyperkalemia, angioedema, renal impairment, hypotension, fetal toxicity (during pregnancy)',
            'contraindications': '- Pregnancy (especially 2nd and 3rd trimester)'
                                 '- Known hypersensitivity'
                                 '- Severe liver dysfunction',
            'Source_of_information': 'Drugs.com, PubChem, BNF'
        },

        {
            'class_name': 'Athenolol',
            'dosage': 'Adults: Starting dose 25–50 mg once daily; maintenance dose 50–100 mg/day. '
                      'Max dose generally 100 mg/day. Adjust for renal impairment.',
            'use': 'Management of hypertension, angina pectoris, post-myocardial infarction care, and certain cardiac arrhythmias',
            'price': 'Approx. Rs. 4–10 per 50 mg tablet, depending on brand and availability in Sri Lanka',
            'side_effects': 'Fatigue, dizziness, cold extremities, slow heart rate, nausea, sleep disturbances',
            'dosage_form': 'Tablets (25 mg, 50 mg, 100 mg); injection form available in hospitals',
            'Scientific_Name': 'Atenolol',
            'max_dose': '100 mg/day (sometimes divided into 50 mg twice daily)',
            'administration': 'Oral; can be taken with or without food; typically once daily at the same time',
            'indications': '- Hypertension\n'
                           '- Angina pectoris\n'
                           '- Post-myocardial infarction\n'
                           '- Supraventricular tachycardia (off-label)\n'
                           '- Migraine prophylaxis (off-label)',
            'precautions': '- Use cautiously in patients with asthma, diabetes, or heart failure\n'
                           '- May mask hypoglycemia symptoms\n'
                           '- Do not discontinue abruptly',
            'serious_effects': 'Bradycardia, hypotension, heart block, bronchospasm, exacerbation of heart failure',
            'contraindications': '- Sinus bradycardia\n'
                                 '- Heart block (2nd or 3rd degree)\n'
                                 '- Cardiogenic shock\n'
                                 '- Overt cardiac failure\n'
                                 '- Hypersensitivity to atenolol',
            'Source_of_information': 'BNF, Drugs.com, MedlinePlus, WHO ATC Index'
        }

    ]
    df = pd.DataFrame(data)
    df.to_csv('Pil-DataSet/drug_information.csv', index=False)
    return df


# Transfer Learning model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# Freeze base model
base_model.trainable = False

# Build model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(full_train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Train the model
history = model.fit(
    full_train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=max(1, len(full_train_generator)),
    validation_steps=max(1, len(val_generator))
)

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('medicine_classifier_model.h5')


# Enhanced prediction function with confidence threshold
def predict_and_show_info(model, image_path, drug_df, threshold=0.7):
    try:
        img = Image.open(image_path).resize((img_height, img_width))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]

        # Get class names
        class_names = {v: k for k, v in full_train_generator.class_indices.items()}
        predicted_class = class_names[class_idx]

        # Check confidence threshold
        if confidence < threshold:
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Not a recognized medicine\n(Confidence: {confidence:.2f})")
            plt.show()
            print("\nThis doesn't appear to be a recognized medicine.")
            print(f"Predicted: {predicted_class} with confidence {confidence:.2f}")
            return

        # Get drug information
        drug_info = drug_df[drug_df['class_name'] == predicted_class].iloc[0]

        # Display results
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}")
        plt.show()

        print("\nDrug Information:")
        print(f"Name: {drug_info['class_name']}")
        print(f"Dosage: {drug_info['dosage']}")
        print(f"Use: {drug_info['use']}")
        print(f"Price: {drug_info['price']}")
        print(f"Side Effects: {drug_info['side_effects']}")

    except Exception as e:
        print(f"Error processing image: {str(e)}")

drug_df = create_drug_info_dataframe()

# Test the function with different types of images
test_images = [
    os.path.join(train_path, "Brufan/broofan.jpeg"),
    os.path.join(train_path,"Paracetamol/ewfef.jpeg"),# Medicine
    os.path.join(train_path, "unknown/susantika.jpeg"),  # Non-medicine
    "paneerbaratha.jpeg"  # Another test image
]

for img_path in test_images:
    if os.path.exists(img_path):
        print(f"\nTesting image: {img_path}")
        predict_and_show_info(model, img_path, drug_df, confidence_threshold)
    else:
        print(f"\nTest image not found: {img_path}")

print("Training completed! Model and drug information saved.")