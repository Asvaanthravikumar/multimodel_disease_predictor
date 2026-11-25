import os
import json
import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from ml_engine.explainability import DiseasePredictor

app = Flask(__name__)
app.secret_key = 'mesopredict-ai-multimodal-disease-prediction-key'

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'patients.json')

# Initialize the ML & SHAP Prediction Engine
predictor = DiseasePredictor()

def load_patients():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading patients: {e}")
    return []

def save_patients(patients):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(patients, f, indent=2)

def get_patient_by_id(patient_id):
    patients = load_patients()
    for p in patients:
        if p['id'] == patient_id:
            return p
    return patients[0] if patients else None

@app.route('/')
def index():
    return render_template('index.html', active_page='home')

@app.route('/patient/new')
def new_patient():
    return render_template('patient_entry.html', active_page='entry')

@app.route('/doctor')
def doctor_default():
    patients = load_patients()
    if not patients:
        return redirect(url_for('new_patient'))
    # Default to first patient (e.g. Rajesh Sharma)
    return redirect(url_for('doctor_portal', patient_id=patients[0]['id']))

@app.route('/doctor/<patient_id>')
def doctor_portal(patient_id):
    patients = load_patients()
    patient = get_patient_by_id(patient_id)
    if not patient and patients:
        patient = patients[0]
    return render_template('doctor_dashboard.html', patient=patient, all_patients=patients, active_page='doctor')

@app.route('/patient')
def patient_default():
    patients = load_patients()
    if not patients:
        return redirect(url_for('new_patient'))
    # Default to first patient
    return redirect(url_for('patient_portal', patient_id=patients[0]['id']))

@app.route('/patient/<patient_id>')
def patient_portal(patient_id):
    patients = load_patients()
    patient = get_patient_by_id(patient_id)
    if not patient and patients:
        patient = patients[0]
    return render_template('patient_dashboard.html', patient=patient, all_patients=patients, active_page='patient')

@app.route('/presentation')
def presentation_view():
    return render_template('presentation_view.html', active_page='presentation')

@app.route('/api/patient/save', methods=['POST'])
def save_patient():
    patients = load_patients()
    new_id = f"P-{1001 + len(patients)}"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    height = float(request.form.get('height', 170))
    weight = float(request.form.get('weight', 70))
    bmi = round(weight / ((height / 100) ** 2), 1)

    patient_record = {
        'id': new_id,
        'name': request.form.get('name', 'Anonymous Patient').strip(),
        'age': int(request.form.get('age', 45)),
        'gender': int(request.form.get('gender', 1)),
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'blood_group': request.form.get('blood_group', 'O+'),
        'smoking': int(request.form.get('smoking', 0)),
        'alcohol': int(request.form.get('alcohol', 0)),
        'physical_activity': int(request.form.get('physical_activity', 1)),
        'sleep_hours': float(request.form.get('sleep_hours', 7.0)),
        'systolic_bp': float(request.form.get('systolic_bp', 120)),
        'diastolic_bp': float(request.form.get('diastolic_bp', 80)),
        'fasting_glucose': float(request.form.get('fasting_glucose', 95)),
        'hba1c': float(request.form.get('hba1c', 5.4)),
        'cholesterol_total': float(request.form.get('cholesterol_total', 185)),
        'cholesterol_hdl': float(request.form.get('cholesterol_hdl', 50)),
        'cholesterol_ldl': float(request.form.get('cholesterol_ldl', 110)),
        'serum_creatinine': float(request.form.get('serum_creatinine', 0.95)),
        'blood_urea': float(request.form.get('blood_urea', 15)),
        'liver_alt': float(request.form.get('liver_alt', 26)),
        'liver_ast': float(request.form.get('liver_ast', 24)),
        'bilirubin_total': float(request.form.get('bilirubin_total', 0.7)),
        'chest_pain': 1 if request.form.get('chest_pain') else 0,
        'shortness_of_breath': 1 if request.form.get('shortness_of_breath') else 0,
        'ecg_abnormal': 1 if request.form.get('ecg_abnormal') else 0,
        'retinal_scan_abnormal': 1 if request.form.get('retinal_scan_abnormal') else 0,
        'registered_at': now_str,
        'notes': request.form.get('notes', 'Initial intake assessment conducted.')
    }

    # Run AI evaluation & SHAP explainability
    assessment = predictor.predict(patient_record)
    patient_record['assessment'] = assessment

    # Prepend new patient so it appears first
    patients.insert(0, patient_record)
    save_patients(patients)

    target = request.form.get('target', 'doctor')
    if target == 'patient':
        return redirect(url_for('patient_portal', patient_id=new_id))
    return redirect(url_for('doctor_portal', patient_id=new_id))

@app.route('/api/patient/notes', methods=['POST'])
def update_patient_notes():
    patient_id = request.form.get('patient_id')
    new_notes = request.form.get('notes', '').strip()
    patients = load_patients()
    for p in patients:
        if p['id'] == patient_id:
            p['notes'] = new_notes
            break
    save_patients(patients)
    return redirect(url_for('doctor_portal', patient_id=patient_id))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    result = predictor.predict(data)
    return jsonify(result)

@app.route('/api/patients', methods=['GET'])
def api_get_patients():
    patients = load_patients()
    return jsonify(patients)

if __name__ == '__main__':
    print("=" * 65)
    print(" MESOPREDICT AI - MULTIMODAL EARLY DISEASE PREDICTION")
    print(" Ensemble Models: Heart, Diabetes, Stroke, CKD, Liver + SHAP XAI")
    print(" Server running on: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host='127.0.0.1', port=5000, debug=True)
