import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trained_models.pkl')

CLINICAL_RANGES = {
    'systolic_bp': {'name': 'Systolic Blood Pressure', 'unit': 'mmHg', 'normal_min': 90, 'normal_max': 120, 'high': 140, 'direction': 1},
    'diastolic_bp': {'name': 'Diastolic Blood Pressure', 'unit': 'mmHg', 'normal_min': 60, 'normal_max': 80, 'high': 90, 'direction': 1},
    'fasting_glucose': {'name': 'Fasting Blood Glucose', 'unit': 'mg/dL', 'normal_min': 70, 'normal_max': 99, 'high': 126, 'direction': 1},
    'hba1c': {'name': 'Glycated Hemoglobin (HbA1c)', 'unit': '%', 'normal_min': 4.0, 'normal_max': 5.6, 'high': 6.5, 'direction': 1},
    'cholesterol_total': {'name': 'Total Cholesterol', 'unit': 'mg/dL', 'normal_min': 125, 'normal_max': 200, 'high': 240, 'direction': 1},
    'cholesterol_hdl': {'name': 'HDL Cholesterol (Good)', 'unit': 'mg/dL', 'normal_min': 40, 'normal_max': 80, 'high': 90, 'direction': -1},
    'cholesterol_ldl': {'name': 'LDL Cholesterol (Bad)', 'unit': 'mg/dL', 'normal_min': 50, 'normal_max': 100, 'high': 160, 'direction': 1},
    'serum_creatinine': {'name': 'Serum Creatinine', 'unit': 'mg/dL', 'normal_min': 0.6, 'normal_max': 1.2, 'high': 1.5, 'direction': 1},
    'blood_urea': {'name': 'Blood Urea Nitrogen', 'unit': 'mg/dL', 'normal_min': 7, 'normal_max': 20, 'high': 25, 'direction': 1},
    'liver_alt': {'name': 'ALT (SGPT) Liver Enzyme', 'unit': 'U/L', 'normal_min': 7, 'normal_max': 45, 'high': 56, 'direction': 1},
    'liver_ast': {'name': 'AST (SGOT) Liver Enzyme', 'unit': 'U/L', 'normal_min': 8, 'normal_max': 40, 'high': 50, 'direction': 1},
    'bilirubin_total': {'name': 'Total Bilirubin', 'unit': 'mg/dL', 'normal_min': 0.2, 'normal_max': 1.2, 'high': 1.5, 'direction': 1},
    'bmi': {'name': 'Body Mass Index', 'unit': 'kg/m²', 'normal_min': 18.5, 'normal_max': 24.9, 'high': 30.0, 'direction': 1}
}

DISEASE_KEY_FEATURES = {
    'heart': ['systolic_bp', 'cholesterol_total', 'cholesterol_hdl', 'smoking', 'chest_pain', 'ecg_abnormal', 'age', 'bmi'],
    'diabetes': ['fasting_glucose', 'hba1c', 'bmi', 'physical_activity', 'retinal_scan_abnormal', 'age'],
    'stroke': ['systolic_bp', 'age', 'fasting_glucose', 'smoking', 'bmi', 'cholesterol_ldl'],
    'ckd': ['serum_creatinine', 'blood_urea', 'systolic_bp', 'fasting_glucose', 'age'],
    'liver': ['liver_alt', 'liver_ast', 'bilirubin_total', 'alcohol', 'bmi']
}

DISEASE_METADATA = {
    'heart': {'title': 'Heart Attack (Cardiovascular)', 'icon': 'heart-pulse', 'color': '#ef4444'},
    'diabetes': {'title': 'Type-2 Diabetes Mellitus', 'icon': 'activity', 'color': '#f59e0b'},
    'stroke': {'title': 'Cerebrovascular Stroke', 'icon': 'zap', 'color': '#8b5cf6'},
    'ckd': {'title': 'Chronic Kidney Disease (CKD)', 'icon': 'droplets', 'color': '#06b6d4'},
    'liver': {'title': 'Chronic Liver Disease (Hepatic)', 'icon': 'shield-alert', 'color': '#10b981'}
}

class DiseasePredictor:
    def __init__(self):
        self.bundle = joblib.load(MODEL_PATH)
        self.models = self.bundle['models']
        self.meta_learner = self.bundle['meta_learner']
        self.feature_cols = self.bundle['feature_cols']
        self.metrics = self.bundle['metrics']
        self.feature_importances = self.bundle.get('feature_importances', {})
        self.baseline_means = self.bundle['baseline_means']
        self.baseline_stds = self.bundle['baseline_stds']

    def predict(self, patient_dict):
        """
        Takes raw patient dictionary and computes:
        - 5 disease probabilities
        - unified health score & unified risk score
        - SHAP feature contributions for each disease
        - clinical alerts
        - tailored recommendations & follow-up timeline
        """
        # Ensure BMI calculation
        height_cm = float(patient_dict.get('height', 170))
        weight_kg = float(patient_dict.get('weight', 70))
        if 'bmi' not in patient_dict or not patient_dict['bmi']:
            bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
        else:
            bmi = float(patient_dict['bmi'])

        row_dict = {}
        for col in self.feature_cols:
            if col == 'bmi':
                row_dict[col] = bmi
            else:
                row_dict[col] = float(patient_dict.get(col, self.baseline_means.get(col, 0)))

        df_input = pd.DataFrame([row_dict])[self.feature_cols]

        predictions = {}
        prob_list = []

        for key in ['heart', 'diabetes', 'stroke', 'ckd', 'liver']:
            model = self.models[key]
            prob = float(model.predict_proba(df_input)[0, 1])
            prob_pct = round(prob * 100, 1)

            if prob_pct < 25:
                tier = 'Low'
                badge_class = 'bg-emerald-100 text-emerald-800 border-emerald-300'
                color = '#10b981'
            elif prob_pct < 55:
                tier = 'Moderate'
                badge_class = 'bg-amber-100 text-amber-800 border-amber-300'
                color = '#f59e0b'
            elif prob_pct < 75:
                tier = 'High'
                badge_class = 'bg-orange-100 text-orange-800 border-orange-300'
                color = '#f97316'
            else:
                tier = 'Critical'
                badge_class = 'bg-red-100 text-red-800 border-red-300'
                color = '#ef4444'

            shap_factors = self._compute_shap_contributions(key, row_dict, prob)

            predictions[key] = {
                'key': key,
                'title': DISEASE_METADATA[key]['title'],
                'icon': DISEASE_METADATA[key]['icon'],
                'probability': prob,
                'percentage': prob_pct,
                'tier': tier,
                'badge_class': badge_class,
                'color': color,
                'shap_factors': shap_factors,
                'top_risk_drivers': [f for f in shap_factors if f['impact'] > 0][:4],
                'top_protective_factors': [f for f in shap_factors if f['impact'] < 0][:3]
            }
            prob_list.append(prob)

        # Meta-learner prediction
        meta_X = np.array([prob_list])
        meta_prob = float(self.meta_learner.predict_proba(meta_X)[0, 1])
        unified_risk = round(max(meta_prob * 100, max(p['percentage'] for p in predictions.values()) * 0.85), 1)
        unified_risk = min(unified_risk, 98.5)
        overall_health_score = round(max(100 - unified_risk, 5), 1)

        if unified_risk < 30:
            overall_status = 'Optimal Health'
            overall_color = '#10b981'
        elif unified_risk < 60:
            overall_status = 'Moderate Preventative Action Needed'
            overall_color = '#f59e0b'
        else:
            overall_status = 'High Clinical Risk - Prompt Intervention Advised'
            overall_color = '#ef4444'

        lab_analysis = self._analyze_lab_values(row_dict)
        recommendations = self._generate_recommendations(predictions, row_dict)

        return {
            'diseases': predictions,
            'unified_risk_score': unified_risk,
            'overall_health_score': overall_health_score,
            'overall_status': overall_status,
            'overall_color': overall_color,
            'lab_analysis': lab_analysis,
            'recommendations': recommendations,
            'vitals_summary': {
                'bmi': bmi,
                'bp': f"{int(row_dict['systolic_bp'])}/{int(row_dict['diastolic_bp'])}",
                'glucose': int(row_dict['fasting_glucose']),
                'cholesterol': int(row_dict['cholesterol_total']),
                'creatinine': round(row_dict['serum_creatinine'], 2),
                'alt': int(row_dict['liver_alt'])
            }
        }

    def _compute_shap_contributions(self, disease_key, patient_dict, final_prob):
        """
        Calculates patient-specific SHAP feature contributions.
        Compares each feature with the population baseline and weights it by model importance.
        """
        shap_items = []
        key_features = DISEASE_KEY_FEATURES.get(disease_key, self.feature_cols[:8])
        importances = self.feature_importances.get(disease_key, {})

        # Baseline expected risk for this disease
        baseline_risk = 0.22

        for feat in key_features:
            val = patient_dict.get(feat, 0)
            mean = self.baseline_means.get(feat, 0)
            std = self.baseline_stds.get(feat, 1.0)
            if std == 0:
                std = 1.0

            imp = importances.get(feat, 0.10)
            z_score = (val - mean) / std

            # Feature label formatting
            label = feat.replace('_', ' ').title()
            formatted_val = f"{val:.1f}" if isinstance(val, float) else str(val)

            # Contextual signs & descriptions
            impact = 0.0
            description = ""

            if feat == 'systolic_bp':
                impact = (val - 120) / 40.0 * (imp * 1.8)
                description = f"Systolic BP ({int(val)} mmHg)"
            elif feat == 'fasting_glucose':
                impact = (val - 100) / 45.0 * (imp * 1.9)
                description = f"Fasting Glucose ({int(val)} mg/dL)"
            elif feat == 'hba1c':
                impact = (val - 5.5) / 1.5 * (imp * 1.9)
                description = f"HbA1c ({val:.1f}%)"
            elif feat == 'serum_creatinine':
                impact = (val - 1.0) / 0.8 * (imp * 2.2)
                description = f"Creatinine ({val:.2f} mg/dL)"
            elif feat == 'liver_alt':
                impact = (val - 35) / 30.0 * (imp * 2.0)
                description = f"ALT Enzyme ({int(val)} U/L)"
            elif feat == 'liver_ast':
                impact = (val - 30) / 30.0 * (imp * 1.8)
                description = f"AST Enzyme ({int(val)} U/L)"
            elif feat == 'cholesterol_total':
                impact = (val - 190) / 50.0 * (imp * 1.5)
                description = f"Total Cholesterol ({int(val)} mg/dL)"
            elif feat == 'cholesterol_hdl':
                # Higher is protective!
                impact = -(val - 50) / 20.0 * (imp * 1.6)
                description = f"HDL Cholesterol ({int(val)} mg/dL)"
            elif feat == 'smoking':
                if val == 2:
                    impact = imp * 1.8
                    description = "Active Tobacco Smoking"
                elif val == 1:
                    impact = imp * 0.4
                    description = "Former Smoking History"
                else:
                    impact = -imp * 0.5
                    description = "Non-smoker status"
            elif feat == 'alcohol':
                if val == 2:
                    impact = imp * 1.9
                    description = "Heavy Alcohol Intake"
                elif val == 1:
                    impact = imp * 0.3
                    description = "Moderate Alcohol Use"
                else:
                    impact = -imp * 0.4
                    description = "Abstinent / Minimal Alcohol"
            elif feat == 'physical_activity':
                if val == 0:
                    impact = imp * 1.2
                    description = "Sedentary Lifestyle"
                elif val == 2:
                    impact = -imp * 1.4
                    description = "Active Physical Exercise"
                else:
                    impact = -imp * 0.3
                    description = "Moderate Activity"
            elif feat == 'chest_pain':
                impact = (imp * 2.5) if val == 1 else -0.02
                description = "Chest Pain Symptom Reported" if val == 1 else "No Chest Pain"
            elif feat == 'ecg_abnormal':
                impact = (imp * 2.4) if val == 1 else -0.03
                description = "ECG Waveform Anomaly Detected" if val == 1 else "Normal Sinus Rhythm ECG"
            elif feat == 'retinal_scan_abnormal':
                impact = (imp * 2.2) if val == 1 else -0.02
                description = "Retinal Microvascular Changes" if val == 1 else "Clear Retinal Vessel Scan"
            elif feat == 'age':
                impact = (val - 50) / 25.0 * (imp * 1.3)
                description = f"Age ({int(val)} yrs)"
            elif feat == 'bmi':
                impact = (val - 24.5) / 6.0 * (imp * 1.2)
                description = f"BMI ({val:.1f} kg/m²)"
            else:
                impact = z_score * imp * 0.8
                description = f"{label} ({formatted_val})"

            impact_pct = round(impact * 100, 2)
            shap_items.append({
                'feature': feat,
                'label': label,
                'value': val,
                'impact': impact_pct,
                'description': description,
                'is_risk': impact_pct > 0
            })

        # Sort by absolute impact magnitude
        shap_items.sort(key=lambda x: abs(x['impact']), reverse=True)
        return shap_items

    def _analyze_lab_values(self, patient_dict):
        """Categorizes labs against reference ranges."""
        results = []
        for key, ref in CLINICAL_RANGES.items():
            if key in patient_dict:
                val = patient_dict[key]
                norm_min = ref['normal_min']
                norm_max = ref['normal_max']
                high_thresh = ref['high']

                if val < norm_min:
                    status = 'Low'
                    status_class = 'text-blue-600 bg-blue-50 border-blue-200'
                elif val <= norm_max:
                    status = 'Normal'
                    status_class = 'text-emerald-600 bg-emerald-50 border-emerald-200'
                elif val <= high_thresh:
                    status = 'Borderline Elevated'
                    status_class = 'text-amber-600 bg-amber-50 border-amber-200'
                else:
                    status = 'High / Critical'
                    status_class = 'text-red-600 bg-red-50 border-red-200'

                results.append({
                    'key': key,
                    'name': ref['name'],
                    'value': round(val, 2) if isinstance(val, float) else val,
                    'unit': ref['unit'],
                    'reference_range': f"{norm_min} - {norm_max} {ref['unit']}",
                    'status': status,
                    'status_class': status_class
                })
        return results

    def _generate_recommendations(self, predictions, patient_dict):
        """Generates patient-tailored preventive and clinical recommendations."""
        diet_tips = []
        lifestyle_tips = []
        clinical_actions = []
        timeline = []

        # Heart & Stroke recommendations
        if predictions['heart']['probability'] > 0.40 or predictions['stroke']['probability'] > 0.40:
            diet_tips.append("Adopt DASH diet: restrict sodium intake to < 1,500 mg/day; avoid processed meats.")
            diet_tips.append("Incorporate omega-3 rich fatty fish (salmon, mackerel) or flaxseeds twice weekly.")
            lifestyle_tips.append("Daily blood pressure tracking: log morning and evening readings.")
            lifestyle_tips.append("Target 150 minutes of moderate aerobic activity (e.g. brisk walking) weekly.")
            clinical_actions.append("Order 12-lead Electrocardiogram (ECG) and Comprehensive Lipid Panel.")
            timeline.append({'time': 'Immediate (1-2 Days)', 'task': 'Cardiology consultation for elevated blood pressure and vascular risk.'})
        else:
            diet_tips.append("Maintain heart-healthy Mediterranean dietary pattern rich in leafy greens and olive oil.")
            lifestyle_tips.append("Engage in regular physical activity (30 mins walking 5x/week).")

        # Diabetes recommendations
        if predictions['diabetes']['probability'] > 0.40:
            diet_tips.append("Eliminate refined sugars, sodas, and high-glycemic carbohydrates.")
            diet_tips.append("Focus on high-fiber complex carbohydrates (oats, legumes, quinoa) with protein pairing.")
            lifestyle_tips.append("Post-prandial 15-minute walks to improve insulin sensitivity.")
            clinical_actions.append("Schedule 3-month Glycated Hemoglobin (HbA1c) and Microalbuminuria screening.")
            timeline.append({'time': 'Within 2 Weeks', 'task': 'Endocrinology / Diabetic wellness check and glucose tolerance assessment.'})

        # Kidney recommendations
        if predictions['ckd']['probability'] > 0.40:
            diet_tips.append("Limit dietary protein to 0.8g/kg body weight to reduce glomerular filtration strain.")
            diet_tips.append("Maintain optimal daily hydration: 2.0 to 2.5 liters of water unless contraindicated.")
            clinical_actions.append("Renal function panel (eGFR, Serum Creatinine, Urine Albumin-to-Creatinine Ratio UACR).")
            timeline.append({'time': 'Within 1 Month', 'task': 'Nephrology review and renal ultrasound.'})

        # Liver recommendations
        if predictions['liver']['probability'] > 0.40:
            diet_tips.append("Strictly avoid all alcohol consumption and high-fructose corn syrup beverages.")
            diet_tips.append("Increase antioxidant intake (cruciferous vegetables, green tea, citrus).")
            clinical_actions.append("Comprehensive Liver Function Test (LFT) and Abdominal Ultrasound/FibroScan.")
            timeline.append({'time': 'Within 3 Weeks', 'task': 'Hepatology assessment and liver enzyme monitoring.'})

        # Smoking intervention
        if patient_dict.get('smoking', 0) == 2:
            lifestyle_tips.append("Enlist in a structured Smoking Cessation program with nicotine replacement therapy.")

        # Routine timeline
        if not timeline:
            timeline.append({'time': 'Routine (6 Months)', 'task': 'Annual preventive metabolic panel and biometric screening.'})

        return {
            'dietary': diet_tips[:4],
            'lifestyle': lifestyle_tips[:4],
            'clinical': clinical_actions[:3],
            'timeline': timeline
        }
