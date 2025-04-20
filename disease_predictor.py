import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import streamlit as st
import pickle
import os
import re
from datetime import datetime

# Download required NLTK data
nltk.download('stopwords')

class DiseasePredictor:
    def __init__(self):
        self.model = None
        self.symptom_encoder = LabelEncoder()
        self.disease_encoder = LabelEncoder()
        self.symptoms = []
        self.diseases = []
        self.disease_descriptions = {
            'Flu': 'A common viral infection that affects the respiratory system.',
            'Malaria': 'A serious disease caused by parasites transmitted through mosquito bites.',
            'Typhoid': 'A bacterial infection that can spread throughout the body.',
            'Dengue': 'A mosquito-borne viral infection that can cause severe flu-like symptoms.',
            'Chicken Pox': 'A highly contagious viral infection causing itchy, blister-like rashes.',
            # Add more diseases and descriptions as needed
        }
        
        # Enhanced disease information database
        self.disease_info = {
            'Flu': {
                'description': 'A common viral infection that affects the respiratory system.',
                'symptoms': ['fever', 'cough', 'sore throat', 'runny nose', 'body aches', 'fatigue'],
                'complications': ['Pneumonia', 'Ear infections', 'Sinus infections', 'Dehydration'],
                'treatments': ['Rest', 'Hydration', 'Over-the-counter medications for fever and pain', 'Antiviral medications in some cases'],
                'prevention': ['Annual flu vaccine', 'Frequent handwashing', 'Avoiding close contact with sick people'],
                'emergency_signs': ['Difficulty breathing', 'Chest pain', 'Sudden dizziness', 'Severe vomiting']
            },
            'Malaria': {
                'description': 'A serious disease caused by parasites transmitted through mosquito bites.',
                'symptoms': ['fever', 'chills', 'headache', 'muscle pain', 'fatigue', 'nausea', 'vomiting'],
                'complications': ['Severe anemia', 'Cerebral malaria', 'Organ failure', 'Low blood sugar'],
                'treatments': ['Antimalarial medications', 'Hospitalization for severe cases', 'Blood transfusions if needed'],
                'prevention': ['Mosquito nets', 'Insect repellents', 'Antimalarial prophylaxis when traveling to endemic areas'],
                'emergency_signs': ['Seizures', 'Difficulty breathing', 'Severe anemia', 'Coma']
            },
            'Typhoid': {
                'description': 'A bacterial infection that can spread throughout the body.',
                'symptoms': ['fever', 'headache', 'abdominal pain', 'constipation', 'diarrhea', 'rash'],
                'complications': ['Intestinal perforation', 'Internal bleeding', 'Encephalitis', 'Pneumonia'],
                'treatments': ['Antibiotics', 'Fluid replacement', 'Surgery in severe cases'],
                'prevention': ['Vaccination', 'Safe food and water practices', 'Good hygiene'],
                'emergency_signs': ['Severe abdominal pain', 'Confusion', 'Difficulty breathing', 'Severe dehydration']
            },
            'Dengue': {
                'description': 'A mosquito-borne viral infection that can cause severe flu-like symptoms.',
                'symptoms': ['fever', 'severe headache', 'joint pain', 'rash', 'fatigue', 'nausea'],
                'complications': ['Dengue hemorrhagic fever', 'Dengue shock syndrome', 'Organ damage'],
                'treatments': ['Supportive care', 'Pain relievers', 'Avoiding aspirin and NSAIDs', 'Hospitalization for severe cases'],
                'prevention': ['Mosquito control', 'Protective clothing', 'Insect repellents'],
                'emergency_signs': ['Severe abdominal pain', 'Persistent vomiting', 'Bleeding gums', 'Rapid breathing']
            },
            'Chicken Pox': {
                'description': 'A highly contagious viral infection causing itchy, blister-like rashes.',
                'symptoms': ['rash', 'itching', 'fever', 'fatigue', 'headache', 'loss of appetite'],
                'complications': ['Bacterial skin infections', 'Pneumonia', 'Encephalitis', 'Reye syndrome'],
                'treatments': ['Calamine lotion', 'Antihistamines', 'Antiviral medications in some cases', 'Pain relievers'],
                'prevention': ['Vaccination', 'Avoiding contact with infected individuals'],
                'emergency_signs': ['Difficulty breathing', 'Confusion', 'Severe headache', 'Rash near eyes']
            }
        }
        
    def preprocess_text(self, text):
        # Simple tokenization using regex
        text = text.lower()
        # Replace underscores with spaces
        text = text.replace('_', ' ')
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+', text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return tokens
    
    def extract_symptoms(self, user_input):
        # Convert user input to symptoms
        tokens = self.preprocess_text(user_input)
        found_symptoms = []
        
        # Simple keyword matching (can be improved with more sophisticated NLP)
        for token in tokens:
            if token in self.symptoms:
                found_symptoms.append(token)
        
        return found_symptoms
    
    def create_symptom_vector(self, symptoms):
        # Create binary vector for symptoms
        vector = np.zeros(len(self.symptoms))
        for symptom in symptoms:
            if symptom in self.symptoms:
                idx = self.symptoms.index(symptom)
                vector[idx] = 1
        return vector
    
    def get_detailed_insights(self, disease, symptoms):
        if disease not in self.disease_info:
            return "Detailed information is not available for this condition."
            
        info = self.disease_info[disease]
        
        # Create a detailed response
        response = f"""
### Detailed Information About {disease}

**Description:**
{info['description']}

**Common Symptoms:**
{', '.join(info['symptoms'])}

**Potential Complications:**
{', '.join(info['complications'])}

**Recommended Treatments:**
{', '.join(info['treatments'])}

**Preventive Measures:**
{', '.join(info['prevention'])}

**When to Seek Emergency Medical Attention:**
{', '.join(info['emergency_signs'])}

**Note:** This information is based on general medical knowledge and may not apply to all cases. Please consult with a healthcare provider for proper diagnosis and treatment.
"""
        return response
    
    def predict_disease(self, symptoms):
        if not symptoms:
            return None, "No symptoms detected. Please describe your symptoms in detail."
        
        vector = self.create_symptom_vector(symptoms)
        if self.model is None:
            return None, "Model not trained yet."
        
        prediction = self.model.predict([vector])[0]
        disease = self.disease_encoder.inverse_transform([prediction])[0]
        
        # Get basic description and recommendations
        description = self.disease_descriptions.get(disease, "No description available.")
        recommendations = self.get_recommendations(disease)
        
        # Get detailed insights from our local database
        detailed_insights = self.get_detailed_insights(disease, symptoms)
        
        return disease, description, recommendations, detailed_insights
    
    def get_recommendations(self, disease):
        # Basic recommendations based on disease
        recommendations = {
            'Flu': "1. Get plenty of rest\n2. Stay hydrated\n3. Take over-the-counter medications for fever and pain\n4. Consult a doctor if symptoms worsen",
            'Malaria': "1. Seek immediate medical attention\n2. Take prescribed antimalarial medications\n3. Use mosquito nets and repellents\n4. Monitor temperature regularly",
            'Typhoid': "1. Seek medical attention\n2. Take prescribed antibiotics\n3. Maintain good hygiene\n4. Follow a light diet",
            'Dengue': "1. Seek medical attention\n2. Stay hydrated\n3. Monitor platelet count\n4. Avoid aspirin and NSAIDs",
            'Chicken Pox': "1. Stay isolated\n2. Keep the rash clean and dry\n3. Take prescribed medications\n4. Avoid scratching the blisters",
        }
        return recommendations.get(disease, "Please consult a healthcare professional for specific recommendations.")

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Disease Prediction Chatbot",
        page_icon="🏥",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stTitle {
            color: #2E4057;
            font-size: 3rem !important;
            padding-bottom: 2rem;
            text-align: center;
        }
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #4A90E2;
            padding: 10px;
        }
        .stButton button {
            border-radius: 20px;
            background-color: #4A90E2;
            color: white;
            padding: 0.5rem 2rem;
            font-size: 1.1rem;
        }
        .stButton button:hover {
            background-color: #357ABD;
            border-color: #357ABD;
        }
        .prediction-box {
            background-color: #F0F8FF;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #4A90E2;
            margin: 10px 0;
        }
        .disclaimer {
            background-color: #FFE5E5;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #FF4444;
            margin-top: 20px;
        }
        .insights-box {
            background-color: #F5F5F5;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid #E0E0E0;
        }
        h3 {
            color: #2E4057;
            border-bottom: 2px solid #4A90E2;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with icon
    st.markdown("<h1 style='text-align: center;'>🏥 Disease Prediction Chatbot with AI Insights</h1>", unsafe_allow_html=True)
    
    # Subheader with description
    st.markdown("""
        <div style='text-align: center; padding: 1rem; margin-bottom: 2rem; color: #666;'>
            Describe your symptoms below and our AI system will help predict possible conditions and provide detailed insights.
        </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Initialize the predictor
        predictor = DiseasePredictor()
        
        # Load the trained model
        try:
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                predictor.model = model_data['model']
                predictor.symptoms = model_data['symptoms']
                predictor.disease_encoder = model_data['disease_encoder']
        except FileNotFoundError:
            st.error("⚠️ Model file not found. Please run train_model.py first.")
            return
        
        # User input section with styling
        st.markdown("<div style='margin-bottom: 1rem;'><b>Enter your symptoms:</b></div>", unsafe_allow_html=True)
        user_input = st.text_area("", height=100, placeholder="Example: I have fever, headache, and body aches...")
        
        # Center the predict button
        col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
        with col1_2:
            predict_button = st.button("🔍 Predict")
    
    with col2:
        # Add some helpful information
        st.markdown("""
            <div style='background-color: #E8F4FE; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h4 style='color: #2E4057;'>💡 Tips for Better Results:</h4>
                <ul style='color: #666;'>
                    <li>Be specific about your symptoms</li>
                    <li>Include duration of symptoms</li>
                    <li>Mention any related conditions</li>
                    <li>Describe symptom severity</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Process prediction
    if predict_button:
        if user_input:
            with st.spinner("🔄 Analyzing symptoms and generating insights..."):
                # Extract symptoms from user input
                symptoms = predictor.extract_symptoms(user_input)
                
                # Get prediction
                disease, description, recommendations, detailed_insights = predictor.predict_disease(symptoms)
                
                if disease:
                    # Display prediction in a styled box
                    st.markdown(f"""
                        <div class='prediction-box'>
                            <h3>🏥 Predicted Condition: {disease}</h3>
                            <p><b>Basic Description:</b> {description}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display detailed insights in a styled box
                    st.markdown("<div class='insights-box'>", unsafe_allow_html=True)
                    st.markdown(detailed_insights)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display recommendations in a styled box
                    st.markdown(f"""
                        <div class='insights-box'>
                            <h3>📋 Basic Recommendations</h3>
                            <pre style='background-color: white; padding: 15px; border-radius: 5px;'>{recommendations}</pre>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add a styled disclaimer
                    st.markdown("""
                        <div class='disclaimer'>
                            <h4 style='color: #CC0000; margin-top: 0;'>⚠️ Medical Disclaimer</h4>
                            <p style='margin-bottom: 0;'>This is an AI-assisted prediction and should not replace professional medical advice. 
                            Please consult with a healthcare provider for proper diagnosis and treatment.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ I couldn't identify any specific symptoms. Please try describing your symptoms in more detail.")
        else:
            st.info("ℹ️ Please enter your symptoms to get a prediction.")

if __name__ == "__main__":
    main() 