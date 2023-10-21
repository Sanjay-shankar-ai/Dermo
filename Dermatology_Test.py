import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the pre-trained model
model = load_model('165class.h5')

# Define a function to preprocess user input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# Define a function for disease prediction
def predict_disease(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Define a dictionary to map class indices to disease names
class_indices_to_diseases = {
    0: 'Acanthosis_nigricans',
    1: 'Acne_Closed_Camedo',
    2: 'Acne_Cystic',
    3: 'Acne_excoriated',
    4: 'Acne_infantile',
    5: 'Acne_open_camedo',
    6: 'Acne_primary_lesion',
    7: 'Acne_pustular',
    8: 'Acne_scar',
    9: 'AIDS',
    10: 'Allergic-contact-dermatitis',
    11: 'Amyloidosis',
    12: 'Angiokeratomas',
    13: 'Angioedema',
    14: 'Atopic Dermatitis',
    15: 'Basal cell carcinoma',
    16: 'Benign_familiar_chronic_pemphigus',
    17: 'Black Heel',
    18: 'Candida_diaper',
    19: 'Candida_groin',
    20: 'Candidiasis_large_skin_fold',
    21: 'Candidiasis_mouth',
    22: 'Candidiasis_penis',
    23: 'Candidiasis_vaginal',
    24: 'Chapped_fissured_feet',
    25: 'Cherry_angioma',
    26: 'Cholinergic_uriticaria',
    27: 'Corns',
    28: 'Cutaneous-Larva-Migrans',
    29: 'Dariers',
    30: 'Dermagraphism',
    31: 'Dermatitis_herpetiformis',
    32: 'Diabetes_mellitus',
    33: 'Diabetic_bullae',
    34: 'Drug-eruption_photosensitivity',
    35: 'Drug-eruptions',
    36: 'Dyshidrosis',
    37: 'Eczema-acute',
    38: 'Eczema_arelo',
    39: 'Eczema-asteatotic',
    40: 'Eczema_arms',
    41: 'Eczema_chronic',
    42: 'Eczema_face',
    43: 'Eczema_hand',
    44: 'Eczema_fingertips',
    45: 'Eczema_foot',
    46: 'Eczema_lids',
    47: 'Eczema_nummular',
    48: 'Eczema_trunk_generalized',
    49: 'Eczema_leg',
    50: 'Erosio-interdigitalis-blastomycetica',
    51: 'Eruptive_xanthoma',
    52: 'Erythema-annulare-centrifugum',
    53: 'Erythema-multiforme',
    54: 'Erythema-nodosum',
    55: 'Ezema-subacute',
    56: 'Gout',
    57: 'Genital_warts',
    58: 'Grovers',
    59: 'Hemangioma',
    60: 'Hemangioma_infancy',
    61: 'Henoch-schonlein-purpura',
    62: 'Herpes-gestations',
    63: 'Herpes-type-1-Primary',
    64: 'Herpes-type-1-Recurrent',
    65: 'Herpes-Zoster',
    66: 'Hidradenitis-suppurativa',
    67: 'Ichthosis',
    68: 'Id_reaction',
    69: 'Interstitial-granulomatous-dermatitis',
    70: 'Impetigo',
    71: 'Hives-urticaria-Acute',
    72: 'Keloids',
    73: 'Keratoacanthoma',
    74: 'Keratolysis-exfoliativa',
    75: 'Lichen planus',
    76: 'Lichen simplex chronicus',
    77: 'Localized_perphigoid',
    78: 'Lyme',
    79: 'Malignant-melanoma',
    80: 'Molluscum-contagiosum',
    81: 'Lymphangioma-circumscriptum',
    82: 'Lupus-chronic-cutaneous',
    83: 'NevoxanthoEndothelioma',
    84: 'Neurofibromatosis',
    85: 'Neurotic_excoriations',
    86: 'Necrobiosis_lipoidica',
    87: 'Nevus_sebaceous',
    88: 'Onycholysis',
    89: 'Porokeratosis',
    90: 'Pompholyx',
    91: 'Perioral-dermatitis',
    92: 'Pemphigus',
    93: 'Pilar cyst',
    94: 'Pemphigus_foliaceous',
    95: 'Pretibial_myxedema',
    96: 'Psoriasis',
    97: 'PUPPP',
    98: 'Puringo_nodularies',
    99: 'Pyoderma_gangrenosum',
    100: 'Pyogenic_granuloma',
    101: 'Rosacea',
    102: 'Sebaceous-hyperplasia',
    103: 'Scabies',
    104: 'Seborrheic_Keratosis_irritated',
    105: 'Schamberg',
    106: 'Seborrheic_keratosis_ruff',
    107: 'Sarcoid',
    108: 'Squamous-cell-carcinoma',
    109: 'Stucco keratoses',
    110: 'Sun-Damaged-Skin',
    111: 'Telangiectasis',
    112: 'Tick-bite',
    113: 'Stasis_dermatitis',
    114: 'Tinea_beard',
    115: 'Tinea_body',
    116: 'Tinea_face',
    117: 'Tinea_foot_dorsum',
    118: 'Tinea_foot_plantar',
    119: 'Tinea_primary_lesion',
    120: 'Tinea_groin',
    121: 'Tinea_foot_webs',
    122: 'Tinea_laboratory',
    123: 'Tinea_hand_dorsum',
    124: 'Tinea_incognito',
    125: 'Tinea_palm',
    126: 'Tinea_scalp',
    127: 'Tinea_versicolor',
    128: 'Tuberous',
    129: 'Vasculitis',
    130: 'Warts',
    131: 'Warts_common',
    132: 'Warts_cryotherapy',
    133: 'Warts_digitate',
    134: 'Warts_flat',
    135: 'Warts_immunocompromised',
    136: 'Warts_plantar',
    137: 'Warts_periungual',
    138: 'Warts_oral',
    139: 'Xanthomas',
    # Continue adding more diseases here
}


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Update the GUI to display the selected image
        image = Image.open(file_path)
        image = image.resize((128, 128), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        # Predict the disease
        predicted_class = predict_disease(file_path)
        predicted_disease_name = class_indices_to_diseases.get(predicted_class, 'Unknown')
        result_label.config(text=f'Predicted Disease: {predicted_disease_name}')

# Create the main GUI window
root = tk.Tk()
root.title("Skin Disease Prediction")

# Create a button to upload an image
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

# Create a label to display the uploaded image
image_label = tk.Label(root)
image_label.pack()

# Create a label to display the predicted disease
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
