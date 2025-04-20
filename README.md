# Disease Prediction Chatbot using SVM

This is a medical assistant chatbot that predicts probable diseases based on user-reported symptoms using a Support Vector Machine (SVM) model.

## Features

- Natural language processing of symptom descriptions
- Disease prediction using SVM
- Detailed disease descriptions and recommendations
- User-friendly web interface using Streamlit

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. First, train the model:
```bash
python train_model.py
```

2. Run the Streamlit app:
```bash
streamlit run disease_predictor.py
```

3. Open your web browser and go to the URL shown in the terminal (usually http://localhost:8501)

4. Enter your symptoms in the text area and click "Predict" to get the diagnosis

## Example Input

Try entering symptoms like:
- "I have a fever, cough, and sore throat"
- "I'm experiencing headache, body aches, and fatigue"
- "I have a rash and fever"

## Important Note

This is a demonstration project and should not be used as a substitute for professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.

## Project Structure

- `disease_predictor.py`: Main application file with the chatbot interface
- `train_model.py`: Script to train the SVM model
- `requirements.txt`: List of required Python packages
- `model.pkl`: Trained model file (generated after running train_model.py)

## Technologies Used

- Python
- scikit-learn (SVM)
- Pandas & NumPy
- NLTK (Natural Language Processing)
- Streamlit (Web Interface) 
