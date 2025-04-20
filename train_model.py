import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

def create_sample_data():
    # Create sample data for demonstration
    data = {
        'symptoms': [
            ['fever', 'cough', 'sore_throat', 'fatigue'],
            ['fever', 'headache', 'body_ache', 'fatigue'],
            ['fever', 'chills', 'sweating', 'headache'],
            ['fever', 'abdominal_pain', 'headache', 'fatigue'],
            ['fever', 'rash', 'headache', 'body_ache'],
            ['rash', 'itching', 'fever', 'fatigue'],
            ['fever', 'cough', 'difficulty_breathing', 'fatigue'],
            ['fever', 'headache', 'stiff_neck', 'sensitivity_to_light'],
            ['fever', 'cough', 'chest_pain', 'difficulty_breathing'],
            ['fever', 'abdominal_pain', 'diarrhea', 'nausea']
        ],
        'disease': [
            'Flu',
            'Flu',
            'Malaria',
            'Typhoid',
            'Dengue',
            'Chicken Pox',
            'Pneumonia',
            'Meningitis',
            'Bronchitis',
            'Gastroenteritis'
        ]
    }
    return pd.DataFrame(data)

def prepare_data(df):
    # Get unique symptoms and diseases
    all_symptoms = set()
    for symptoms in df['symptoms']:
        all_symptoms.update(symptoms)
    
    symptoms_list = sorted(list(all_symptoms))
    diseases_list = sorted(df['disease'].unique())
    
    # Create binary feature matrix
    X = np.zeros((len(df), len(symptoms_list)))
    for i, symptoms in enumerate(df['symptoms']):
        for j, symptom in enumerate(symptoms_list):
            if symptom in symptoms:
                X[i, j] = 1
    
    # Encode diseases
    le = LabelEncoder()
    y = le.fit_transform(df['disease'])
    
    return X, y, symptoms_list, diseases_list, le

def train_model():
    # Create and prepare data
    df = create_sample_data()
    X, y, symptoms_list, diseases_list, disease_encoder = prepare_data(df)
    
    # Train SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    
    # Save model and encoders
    with open('model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'symptoms': symptoms_list,
            'disease_encoder': disease_encoder
        }, f)
    
    print("Model trained and saved successfully!")
    print(f"Number of symptoms: {len(symptoms_list)}")
    print(f"Number of diseases: {len(diseases_list)}")

if __name__ == "__main__":
    train_model() 