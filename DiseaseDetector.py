import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import shutil

# Dataset paths
train_path = "Disease/train"
val_path = "Disease/validation"
unknown_path = "Disease/unknown"

# Create directories if they don't exist
os.makedirs(os.path.join(train_path, "unknown"), exist_ok=True)
os.makedirs(os.path.join(val_path, "unknown"), exist_ok=True)

# Add some non-disease images to the unknown class
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

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Fixed typo from 'resclase'
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Training generator
full_train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
)

# Validation generator (should point to val_path, not train_path)
val_generator = train_datagen.flow_from_directory(
    val_path,  # Changed from train_path to val_path
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
)

def create_disease_info_dataframe():
    data = [

        {
            'Disease': 'Eczema',
            'Description': 'A chronic inflammatory skin condition characterized by itchy, red, and dry skin.',
            'symptoms': 'Itching, redness, dry skin, inflammation, rashes, cracked or scaly skin',
            'cause': 'Genetic and environmental factors; often linked to allergies or immune system dysfunction',
            'side_effects': 'Skin infections, sleep disturbances, thickened skin from scratching',
            'treatment': 'Moisturizers, topical corticosteroids, antihistamines, immunosuppressants',
            'medications': 'Hydrocortisone, Tacrolimus, Antihistamines, Dupilumab',
            'prevention': 'Avoid known triggers, use gentle skin care products, keep skin moisturized',
            'severity': 'Varies from mild to severe and chronic',
            'risk_factors': 'Family history, allergies, asthma, environmental irritants',
            'is_contagious': 'No',
            'common_age_group': 'Infants, children, but can occur at any age',
            'duration': 'Chronic (long-term); symptoms can flare up periodically',
            'first_aid_advice': 'Apply moisturizer, avoid scratching, use prescribed creams',
            'Source_of_information': 'WHO, Mayo Clinic, American Academy of Dermatology',
            'scientific_name': 'Atopic Dermatitis'
        },

        {
            'Disease': 'Chickenpox',
            'Description': 'Contagious viral infection causing itchy rash and blisters',
            'symptoms': 'Fever, fatigue, itchy rash ith red spots and blisters',
            'cause': 'Varicella-zoster virus',
            'side_effects': 'Skin Infections, pneumonia, encephalitis',
            'treatment': 'Antiviral medications, calamine lotion, antihistamines',
            'medications': 'Acyclovir, calamine lotion, diphenhydramine',
            'prevention': 'Varicella vaccine',
            'severity': 'Moderate',
            'risk_factors': 'Unvaccinated individuals',
            'is_contagious': 'Yes',
            'common_age_group': 'Children',
            'duration': '1-2 weeks',
            'first_aid_advice': 'keep nails short, apply, soothing lotions, isolate',
            'Source_of_information': 'CDC - Chickenpox',
            'scientific_name': 'Varicella'
        },

        {
            'Disease': 'Psoriasis',
            'Description': 'Autoimmune condition using rapid skin cell buildup',
            'symptoms': 'Scaly, red patches with silvery scales',
            'cause': 'Autoimmune response',
            'side_effects': 'Itching, burning, joint pain (psoriatic arthritis)',
            'treatment': 'Topical creams, phototherapy, biologics.',
            'medications': 'Corticosteroids, Methotrexate, Biologics (e.g. Adalimumab)',
            'prevention': 'Stress reduction, skin cre, avoid triggers',
            'severity': 'Mild to debilitating',
            'risk_factors': 'Family history, infection, stress',
            'is_contagious': 'No',
            'common_age_group': '15-35 years',
            'duration': 'Lifelong with flare-ups',
            'first_aid_advice': 'Use moisturizers and anti-inflammatory creams',
            'Source_of_information': 'National Psoriasis Foundation',
            'scientific_name': 'Psoriasis vulgaris'
        },


        {
            'Disease': 'Ringworm',
            'Description': 'Fungal infection forming circular red rashes.',
            'symptoms': 'Circular red rash, itching, scaly skin.',
            'cause': 'Dermatophyte fungi.',
            'side_effects': 'Skin cracks, secondary infection.',
            'treatment': ' Antifungal creams or oral medication.',
            'medications': 'Clotrimazole, Terbinafine, Griseofulvin',
            'prevention': 'Good hygiene, avoid sharing personal items.',
            'severity': 'Mild to Moderate',
            'risk_factors': 'Damp environment, skin contact sports.',
            'is_contagious': 'Yes',
            'common_age_group': 'Children, ahtletes',
            'duration': '2-4 weeks',
            'first_aid_advice': 'Apply antifungal, keep dry and clean.',
            'Source_of_information': 'Cleveland Clinic - Ringworm',
            'scientific_name': 'Tinea corporis'
        },

        {
            'Disease': 'Rosacea',
            'Description': 'Chronic facial skin condition with redness and visible blood vessels.',
            'symptoms': 'Redness, swelling, visible blood vessels, acne-like breakouts.',
            'cause': 'Unknown; possibly immune, environmental, or vascular factors.',
            'side_effects': 'Eye irritation, thickened skin.',
            'treatment': 'Topical antibiotics, laser therapy.',
            'medications': 'Metronidazole, Azelaic acid, Doxycycline',
            'prevention': 'Avoid heat, spicy food, alcohol.',
            'severity': 'Chronic but manageable.',
            'risk_factors': 'Fair skin, family history.',
            'is_contagious': ' No.',
            'common_age_group': '30–50 years.',
            'duration': 'Long-term.',
            'first_aid_advice': 'Avoid triggers, use gentle cleansers.',
            'Source_of_information': 'National Rosacea Society',
            'scientific_name': 'Rosacea'
        },

        {
            'Disease': 'Impetigo',
            'Description': 'Highly contagious bacterial skin infection.',
            'symptoms': 'Red sores that burst and form honey-colored crust.',
            'cause': 'Staph or Strep bacteria.',
            'side_effects': 'Cellulitis, scarring.',
            'treatment': 'Antibiotic cream or oral antibiotics.',
            'medications': 'Mupirocin, Cephalexin, Amoxicillin',
            'prevention': 'Hygiene, avoid sharing items.',
            'severity': 'Mild to moderate.',
            'risk_factors': 'Warm climates, young children.',
            'is_contagious': 'Yes.',
            'common_age_group': 'Children (2–5 years).',
            'duration': '7–10 days with treatment.',
            'first_aid_advice': 'Clean with mild soap, cover with gauze, avoid scratching.',
            'Source_of_information': 'NHS - Impetigo',
            'scientific_name': 'Impetigo contagiosa'
        },

        {
            'Disease': 'Cataract',
            'Description': 'Clouding of the lens in the eye leading to decreased vision.',
            'symptoms': 'Blurry vision, difficulty seeing at night, sensitivity to light, seeing halos around lights.',
            'cause': 'Aging, diabetes, smoking, prolonged exposure to sunlight.',
            'side_effects': 'Vision impairment, blindness if untreated.',
            'treatment': 'Surgical removal of the clouded lens and replacement with an artificial lens.',
            'medications': 'No medications can cure cataracts, but eye drops may help manage symptoms temporarily.',
            'prevention': 'Wearing sunglasses, managing diabetes, avoiding smoking, regular eye exams.',
            'severity': 'Progressive and can lead to blindness if untreated.',
            'risk_factors': 'Age, UV exposure, smoking, diabetes, eye injury.',
            'is_contagious': 'No',
            'common_age_group': 'People over 60 years old.',
            'duration': 'Progressive; worsens over months or years.',
            'first_aid_advice': 'Protect the eyes from further damage; seek an ophthalmologist.',
            'Source_of_information': 'WHO, Mayo Clinic, WebMD'
        },

        {
            'Disease': 'Conjunctivitis',
            'Description': 'Inflammation of the conjunctiva (the membrane covering the white part of the eye).',
            'symptoms': 'Redness, itching, tearing, discharge, gritty feeling in the eye.',
            'cause': 'Viral or bacterial infection, allergies, irritants like smoke or chlorine.',
            'side_effects': 'Discomfort, blurred vision, spread of infection.',
            'treatment': 'Antibiotic or antiviral eye drops (for infections), antihistamines (for allergies).',
            'medications': 'Tobramycin, Ciprofloxacin, Olopatadine.',
            'prevention': 'Good hygiene, avoiding sharing towels or cosmetics, avoiding allergens.',
            'severity': 'Usually mild but can be severe in some infections.',
            'risk_factors': 'Close contact with infected individuals, allergies, poor hygiene.',
            'is_contagious': 'Yes, if caused by bacteria or virus.',
            'common_age_group': 'Children and adults in close-contact settings.',
            'duration': '3–7 days (viral), up to 2 weeks (bacterial).',
            'first_aid_advice': 'Avoid touching the eyes, clean discharge, apply prescribed drops.',
            'Source_of_information': 'CDC, American Academy of Ophthalmology'
        },

        {
            'Disease': 'Stye',
            'Description': 'A red, painful lump near the edge of the eyelid caused by a bacterial infection.',
            'symptoms': 'Swelling, redness, pain, tenderness, teariness.',
            'cause': 'Bacterial infection (usually Staphylococcus) of oil glands in the eyelid.',
            'side_effects': 'Temporary blurred vision, pain, discomfort.',
            'treatment': 'Warm compresses, antibiotic ointments or drops if needed.',
            'medications': 'Erythromycin ointment, warm saline solution.',
            'prevention': 'Keep eyelids clean, avoid sharing cosmetics, replace old makeup.',
            'severity': 'Mild, self-limiting in most cases.',
            'risk_factors': 'Poor eyelid hygiene, rubbing eyes with dirty hands, use of old cosmetics.',
            'is_contagious': 'No, but the bacteria causing it can spread.',
            'common_age_group': 'All age groups, more common in teenagers and adults.',
            'duration': '3–7 days typically.',
            'first_aid_advice': 'Apply warm compresses for 10-15 minutes several times a day.',
            'Source_of_information': 'Mayo Clinic, NHS'
        },

        {
            'Disease': 'Herpes Simplex',
            'Description': 'An eye infection caused by the herpes simplex virus (HSV), affecting the cornea.',
            'symptoms': 'Eye redness, pain, blurred vision, watery discharge, sensitivity to light.',
            'cause': 'Herpes Simplex Virus Type 1 (HSV-1), usually from a cold sore virus.',
            'side_effects': 'Corneal scarring, vision loss, recurrent infections.',
            'treatment': 'Antiviral eye drops or oral medications.',
            'medications': 'Acyclovir, Trifluridine drops.',
            'prevention': 'Avoid touching the eyes after touching cold sores, proper hygiene.',
            'severity': 'Can be serious and lead to blindness if untreated.',
            'risk_factors': 'Previous HSV infection, immunosuppression.',
            'is_contagious': 'Yes, HSV is contagious.',
            'common_age_group': 'Young adults to middle-aged individuals.',
            'duration': '7–14 days (initial), can recur.',
            'first_aid_advice': 'Avoid touching the infected eye, seek medical treatment immediately.',
            'Source_of_information': 'American Academy of Ophthalmology, CDC'
        },

        {
            'Disease': 'Canker Sores',
            'Description': 'Small, shallow ulcers that develop on the soft tissues inside the mouth or at the base of the gums.',
            'symptoms': 'Painful sores inside the mouth, tingling or burning sensation before the sores appear, difficulty eating or talking',
            'cause': 'Minor injury to the mouth, stress, acidic or spicy foods, vitamin deficiencies (B12, iron), hormonal changes',
            'side_effects': 'Pain, discomfort while eating or speaking, possible secondary infection if severe',
            'treatment': 'Topical pastes, mouth rinses, pain relievers, avoiding trigger foods',
            'medications': 'Benzocaine, Fluocinonide, Hydrogen peroxide rinses',
            'prevention': 'Avoid trigger foods, maintain good oral hygiene, reduce stress, use soft-bristled toothbrushes',
            'severity': 'Usually mild and self-limiting',
            'risk_factors': 'Family history, stress, dietary deficiencies, certain food sensitivities',
            'is_contagious': 'No',
            'common_age_group': 'Teens and young adults, but can occur at any age',
            'duration': '7–14 days without treatment',
            'first_aid_advice': 'Rinse mouth with salt water or baking soda, apply topical pain relievers, avoid spicy or acidic foods',
            'Source_of_information': 'Mayo Clinic, WebMD, American Dental Association'
        },

        {
            'Disease': 'Nail Fungus',
            'Description': 'A fungal infection that affects the fingernails or toenails, often causing discoloration, thickening, and crumbling.',
            'symptoms': 'Discolored nails, thickened nails, brittle or crumbly nails, distorted nail shape, foul smell',
            'cause': 'Fungal organisms (mainly dermatophytes), moist environments, poor foot hygiene',
            'side_effects': 'Pain, permanent nail damage, spread of infection to other nails or skin',
            'treatment': 'Antifungal medications, topical treatments, nail removal in severe cases',
            'medications': 'Terbinafine, Itraconazole, Ciclopirox, Efinaconazole',
            'prevention': 'Keep feet dry and clean, avoid walking barefoot in communal showers, wear breathable footwear, don’t share nail clippers',
            'severity': 'Mild to moderate, can become chronic if untreated',
            'risk_factors': 'Age, sweating heavily, working in humid environments, wearing tight shoes, compromised immunity',
            'is_contagious': 'Yes, through direct contact or shared surfaces',
            'common_age_group': 'Adults, especially older adults',
            'duration': 'Several months; full treatment can take up to 12 months',
            'first_aid_advice': 'Keep affected nails dry and trimmed, apply antifungal cream, avoid spreading infection',
            'Source_of_information': 'CDC, Mayo Clinic, American Academy of Dermatology'
        }
    ]

    df = pd.DataFrame(data)
    df.to_csv('Disease/disease_information.csv', index=False)
    return df

# Transfer Learning model
base_model = MobileNetV2(
    weights='imagenet',  # Fixed typo from 'imagent'
    include_top=False,  # Changed from string 'False' to boolean False
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
    metrics=['accuracy']  # Fixed typo from 'metrices'
)

# Model summary
model.summary()

# Train the model
history = model.fit(
    full_train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=max(1, len(full_train_generator)),
    validation_steps=max(1, len(val_generator))  # Fixed typo from 'validaton_steps'
)

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()  # Fixed typo from 'legent'

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
model.save('disease_classifier_model.h5')

def predict_and_show_info(model, image_path, disease_df, threshold=0.7):  # Fixed function name typo
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
            plt.title(f"Not a recognized disease\n(Confidence: {confidence:.2f})")
            plt.show()
            print("\nThis doesn't appear to be a recognized disease.")
            print(f"Predicted: {predicted_class} with confidence {confidence:.2f}")
            return

        # Get disease information
        disease_info = disease_df[disease_df['Disease'] == predicted_class].iloc[0]

        # Display results
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}")
        plt.show()

        print("\nDisease Information:")
        print(f"Disease: {disease_info['Disease']}")
        print(f"Description: {disease_info['Description']}")
        print(f"Symptoms: {disease_info['symptoms']}")
        print(f"Treatment: {disease_info['treatment']}")

    except Exception as e:
        print(f"Error processing image: {str(e)}")

disease_df = create_disease_info_dataframe()

# Test the function with different types of images
test_images = [
    os.path.join(val_path,"Eczema/a.jpg"),
    os.path.join(val_path,"Chickenpox/a.jpeg"),
    os.path.join(val_path, "Impetigo/a.jpg"),
    os.path.join(val_path, "Psoriasis/a.jpeg"),
    os.path.join(val_path, "Ringworm/a.jpeg"),
    os.path.join(val_path, "Rosacea/a.jpeg"),
    os.path.join(val_path,"Cataract/a.jpeg"),
    os.path.join(val_path,"Conjunctivitis/a.jpg"),
    os.path.join(val_path,"Stye/a.jpeg"),
    os.path.join(val_path,"Herpes Simplex/a.jpeg"),
    os.path.join(val_path,"Canker Sores/a.jpeg"),
    os.path.join(val_path,"Nail Fungus/a.jpg")

]

for img_path in test_images:
    if os.path.exists(img_path):
        print(f"\nTesting image: {img_path}")
        predict_and_show_info(model, img_path, disease_df, confidence_threshold)
    else:
        print(f"\nTest image not found: {img_path}")

print("Training completed! Model and disease information saved.")