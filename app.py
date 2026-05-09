# ============================================================
# FLASK DEPLOYMENT - POSTPARTUM MENTAL HEALTH RISK PREDICTOR
# EXACT ORIGINAL LABELS DISPLAYED
# ============================================================

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = './models'

# Load all models and preprocessors
print("="*50)
print("LOADING MODELS...")
print("="*50)

anxiety_model = joblib.load(os.path.join(MODEL_PATH, 'anxiety_model_best.pkl'))
print("✓ Anxiety Model loaded (Random Forest)")

suicide_model = joblib.load(os.path.join(MODEL_PATH, 'suicide_model.pkl'))
print("✓ Suicide Model loaded (Random Forest)")

scaler_anxiety = joblib.load(os.path.join(MODEL_PATH, 'scaler_anxiety.pkl'))
scaler_suicide = joblib.load(os.path.join(MODEL_PATH, 'scaler_suicide.pkl'))
print("✓ Scalers loaded")

feature_cols = joblib.load(os.path.join(MODEL_PATH, 'feature_columns.pkl'))
risk_thresholds = joblib.load(os.path.join(MODEL_PATH, 'risk_thresholds.pkl'))

# Load label mappings
label_mappings = joblib.load(os.path.join(MODEL_PATH, 'label_mappings.pkl'))
ANXIETY_LABELS = label_mappings['anxiety']  # {0: 'NO ANXIETY', 1: 'ANXIETY (YES)'}
SUICIDE_LABELS = label_mappings['suicide']  # {0: 'NO', 1: 'YES', 2: 'NOT INTERESTED TO SAY'}

print("\n" + "="*50)
print("EXACT LABELS LOADED:")
print(f"  Anxiety: {ANXIETY_LABELS}")
print(f"  Suicide: {SUICIDE_LABELS}")
print("="*50)

# Symptom mapping for API inputs
SYMPTOM_MAPPING = {
    'sad': {'yes': 1, 'no': 0, 'sometimes': 0.5},
    'irritable': {'yes': 1, 'no': 0, 'sometimes': 0.5},
    'sleep': {'yes': 1, 'no': 0, 'two or more days a week': 1},
    'concentration': {'yes': 1, 'no': 0, 'often': 1},
    'appetite': {'yes': 1, 'no': 0, 'not at all': 0},
    'guilt': {'yes': 1, 'no': 0, 'maybe': 0.5},
    'bonding': {'yes': 1, 'no': 0, 'sometimes': 0.5}
}

def parse_symptom_value(symptom_name, value):
    if value is None:
        return 0
    value_str = str(value).lower().strip()
    mapping = SYMPTOM_MAPPING.get(symptom_name, {})
    return mapping.get(value_str, 0)

def get_anxiety_risk_level(probability):
    if probability > risk_thresholds['severe_risk']:
        return 'SEVERE RISK', '#dc3545', 'Immediate psychiatric evaluation required', 5
    elif probability > risk_thresholds['high_risk']:
        return 'HIGH RISK', '#fd7e14', 'Schedule appointment within 1 week', 4
    elif probability > risk_thresholds['moderate_risk']:
        return 'MODERATE RISK', '#ffc107', 'Monitor closely, follow-up in 2 weeks', 3
    elif probability > risk_thresholds['low_risk']:
        return 'LOW RISK', '#28a745', 'Routine monitoring at next visit', 2
    else:
        return 'MINIMAL RISK', '#17a2b8', 'Continue standard postpartum care', 1

def get_suicide_risk_level(prediction):
    if prediction == 1:
        return 'YES', '#dc3545', 'URGENT: Immediate mental health assessment required', 5
    elif prediction == 2:
        return 'NOT INTERESTED TO SAY', '#ffc107', 'Further assessment needed - patient reluctant to disclose', 3
    else:
        return 'NO', '#28a745', 'Continue routine monitoring', 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models': {
            'anxiety': 'Random Forest (Best CV AUC: 0.943)',
            'suicide': 'Random Forest (Accuracy: 73.1%)'
        },
        'label_mappings': {
            'anxiety': ANXIETY_LABELS,
            'suicide': SUICIDE_LABELS
        },
        'data_quality_note': 'Potential coding reversal detected in anxiety labels - documented'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        patient_data = {
            'Age': float(data.get('age', 32)),
            'Sad': parse_symptom_value('sad', data.get('sad')),
            'Irritable': parse_symptom_value('irritable', data.get('irritable')),
            'Sleep': parse_symptom_value('sleep', data.get('sleep')),
            'Concentration': parse_symptom_value('concentration', data.get('concentration')),
            'Appetite': parse_symptom_value('appetite', data.get('appetite')),
            'Guilt': parse_symptom_value('guilt', data.get('guilt')),
            'Bonding': parse_symptom_value('bonding', data.get('bonding')),
            'Hour': int(data.get('hour', 14))
        }
        
        patient_df = pd.DataFrame([patient_data])[feature_cols]
        
        patient_scaled_anxiety = scaler_anxiety.transform(patient_df)
        patient_scaled_suicide = scaler_suicide.transform(patient_df)
        
        # ANXIETY PREDICTION
        anxiety_proba = float(anxiety_model.predict_proba(patient_scaled_anxiety)[0][1])
        anxiety_prediction_code = 1 if anxiety_proba >= 0.5 else 0
        anxiety_exact_label = ANXIETY_LABELS[anxiety_prediction_code]
        anxiety_level, anxiety_color, anxiety_recommendation, anxiety_score = get_anxiety_risk_level(anxiety_proba)
        
        # SUICIDE PREDICTION
        suicide_prediction_code = int(suicide_model.predict(patient_scaled_suicide)[0])
        suicide_exact_label = SUICIDE_LABELS[suicide_prediction_code]
        suicide_proba = suicide_model.predict_proba(patient_scaled_suicide)[0].tolist()
        suicide_level, suicide_color, suicide_recommendation, suicide_score = get_suicide_risk_level(suicide_prediction_code)
        
        # OVERALL RISK
        overall_score = max(anxiety_score, suicide_score)
        if overall_score >= 4:
            overall_risk = 'HIGH RISK'
            overall_color = '#dc3545'
            overall_action = 'Immediate clinical attention required'
        elif overall_score >= 3:
            overall_risk = 'MODERATE RISK'
            overall_color = '#ffc107'
            overall_action = 'Schedule follow-up appointment within 2 weeks'
        elif overall_score >= 2:
            overall_risk = 'LOW RISK'
            overall_color = '#28a745'
            overall_action = 'Monitor at next routine visit'
        else:
            overall_risk = 'MINIMAL RISK'
            overall_color = '#17a2b8'
            overall_action = 'Continue standard postpartum care'
        
        symptom_cols = ['Sad', 'Irritable', 'Sleep', 'Concentration', 'Appetite', 'Guilt', 'Bonding']
        symptom_count = sum([patient_data[col] for col in symptom_cols])
        
        return jsonify({
            'success': True,
            'patient_data': {
                'age': patient_data['Age'],
                'symptom_count': symptom_count,
                'hour': patient_data['Hour']
            },
            'anxiety': {
                'probability': anxiety_proba,
                'probability_percent': f"{anxiety_proba*100:.1f}%",
                'exact_label': anxiety_exact_label,
                'prediction_code': anxiety_prediction_code,
                'risk_level': anxiety_level,
                'color': anxiety_color,
                'recommendation': anxiety_recommendation,
                'score': anxiety_score
            },
            'suicide': {
                'exact_label': suicide_exact_label,
                'prediction_code': suicide_prediction_code,
                'probabilities': {
                    'no': round(suicide_proba[0], 4) if len(suicide_proba) > 0 else 0,
                    'yes': round(suicide_proba[1], 4) if len(suicide_proba) > 1 else 0,
                    'not_interested': round(suicide_proba[2], 4) if len(suicide_proba) > 2 else 0
                },
                'risk_level': suicide_level,
                'color': suicide_color,
                'recommendation': suicide_recommendation,
                'score': suicide_score
            },
            'overall_risk': {
                'level': overall_risk,
                'color': overall_color,
                'action': overall_action,
                'score': overall_score
            },
            'disclaimer': 'Model trained on original labels. Potential coding reversal documented.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)