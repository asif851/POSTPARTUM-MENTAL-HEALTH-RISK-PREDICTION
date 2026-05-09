# Postpartum Mental Health Risk Prediction System

A machine learning-based web application for screening postpartum depression, anxiety, and suicide risk in new mothers.

## Overview

This system uses clinical survey data to predict:
- Anxiety risk (Binary: Yes/No)
- Suicide risk (Multi-class: No/Yes/Not interested to say)

The application provides healthcare professionals with a non-invasive screening tool to identify mothers who may need mental health support.

## Dataset

The model was trained on 1,503 patient records with the following features:

### Input Features
- Age range (25-30, 30-35, 35-40, 40-45, 50+)
- Feeling sad or tearful (No, Sometimes, Yes)
- Irritable towards baby and partner (No, Sometimes, Yes)
- Trouble sleeping at night (No, Yes, Two or more days a week)
- Problems concentrating or making decisions (No, Yes, Often)
- Overeating or loss of appetite (No, Yes, Not at all)
- Feeling of guilt (No, Yes, Maybe)
- Problems of bonding with baby (No, Sometimes, Yes)

### Target Variables
- Feeling anxious (Yes/No)
- Suicide attempt (No/Yes/Not interested to say)

## Model Performance

### Anxiety Model (Binary Classification)
- Algorithm: Gradient Boosting Classifier
- ROC-AUC: 0.9758
- Accuracy: 92.03%
- Precision: 0.92
- Recall: 0.92
- F1-Score: 0.92
- Optimal Threshold: 0.3

### Suicide Risk Model (Multi-Class Classification)
- Algorithm: Gradient Boosting Classifier
- ROC-AUC: 0.9355
- Accuracy: 79.07%
- Weighted Precision: 0.79
- Weighted Recall: 0.79
- Weighted F1-Score: 0.79

### Cross-Validation Results
- Anxiety 5-fold CV ROC-AUC: 0.9737 (+/- 0.0055)
- Suicide 5-fold CV Accuracy: 0.785 (+/- 0.0096)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone or download the project

2. Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Place model files in the ./models/ directory
   Required files:
   - anxiety_model.pkl
   - suicide_model.pkl
   - feature_columns.pkl
   - symptom_mappings.pkl
   - metadata.pkl

5. Run the application
   python app.py

6. Access the application
   Open your browser and navigate to: http://127.0.0.1:5000

## Project Structure

postpartum/
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface
├── models/               # Trained model files
│   ├── anxiety_model.pkl
│   ├── suicide_model.pkl
│   ├── feature_columns.pkl
│   ├── symptom_mappings.pkl
│   └── metadata.pkl
└── README.md

## API Endpoints

- GET / : Main web interface
- POST /predict : Submit symptoms and get predictions
- GET /debug : Debug endpoint to verify model health

## Clinical Validation

### Sanity Check Results
For a patient with all "No" symptoms:
- Anxiety probability: 15.5% (Expected: <30%)
- Prediction: No Anxiety (Correct)
- Suicide prediction: No (Correct)

### Test Scenarios

- Healthy: None (all "No") -> No Anxiety (15.5%) -> No
- Mild: Sad, Guilt -> Yes Anxiety (34.5%) -> No
- Moderate: 4 symptoms -> Yes Anxiety (75.3%) -> Yes
- Severe: 7 symptoms -> Yes Anxiety (83.9%) -> Yes

## Output Interpretation

### Anxiety Results
- Yes: High probability of anxiety symptoms; clinical evaluation recommended
- No: Low probability of anxiety; routine monitoring sufficient

### Suicide Risk Results
- Yes: High suicide risk; immediate mental health intervention needed
- No: No identified suicide risk; continue standard care
- Not interested to say: Patient declined to answer; further assessment may be needed

## Disclaimer

This tool is for screening purposes only. It is not a substitute for professional medical diagnosis. Always consult with qualified healthcare providers for clinical decisions. If you are experiencing thoughts of harming yourself or others, please seek immediate help from a mental health professional or call a crisis hotline.

## License

This project is for research and clinical screening purposes.

## Contact

For questions or support, please contact the development team.
# postpartum-mental-health-predictor
# POSTPARTUM-MENTAL-HEALTH-RISK-PREDICTION
