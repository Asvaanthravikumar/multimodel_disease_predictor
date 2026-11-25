# MesoPredict AI - Multimodal Early Disease Prediction System
**Clinical Preventive Intelligence & Decision Support Platform**

---

## 🌟 Key Innovations & System Features

1. **Multimodal Clinical Input Fusion**:
   - Demographics & Biometrics (Age, Gender, Height, Weight, Auto BMI)
   - Behavioral & Lifestyle (Smoking history, Alcohol, Activity level, Sleep hours)
   - Cardiometabolic Blood Chemistry (BP, Fasting Glucose, HbA1c, Lipids)
   - Renal & Hepatic Biomarkers (Serum Creatinine, Blood Urea, ALT/SGPT, AST/SGOT, Bilirubin)
   - Symptoms Matrix & Multimodal Medical Scan features (ECG segment anomaly, Retinal fundus scan)

2. **Parallel Ensemble Machine Learning & Stacking Meta-Learner**:
   - **Heart Attack Risk Model**: 93.83% Accuracy, 0.9821 ROC-AUC
   - **Type-2 Diabetes Risk Model**: 90.50% Accuracy, 0.9578 ROC-AUC
   - **Stroke Prediction Model**: 96.17% Accuracy, 0.9720 ROC-AUC
   - **Chronic Kidney Disease (CKD) Model**: 96.17% Accuracy, 0.9923 ROC-AUC
   - **Liver Disease Model**: 93.50% Accuracy, 0.9854 ROC-AUC
   - **Stacking Meta-Learner**: Combines predictions into a single Unified Health Risk Index (0–100) and Health Index Score.

3. **SHAP (Shapley Additive exPlanations) Explainable AI**:
   - Calculates exact positive risk weights and protective factors for each patient.
   - Interactive waterfall and horizontal bar charts (Chart.js) explaining *why* the AI predicted a given risk.

4. **Dual Dedicated Portals**:
   - **Doctor Portal (`/doctor`)**: Clinical triage table, 5-disease probabilistic breakdown, SHAP waterfall charts, laboratory reference ranges, clinical impressions notes, and 1-click printable clinical diagnostic report.
   - **Patient Portal (`/patient`)**: Accessible 0-100 Health Score dial, plain-language risk cards, personalized dietary guidelines (DASH diet, low glycemic), physical exercise routine, and care follow-up calendar.
   - **Intake Form (`/patient/new`)**: Includes **1-Click Demo Personas** for live presentations.
   - **Architecture & Review View (`/presentation`)**: Digital interactive version of System Architecture flowchart and Literature Review matrix.

---

## 🚀 How to Run the Project

### Option A: 1-Click Launch (Windows)
Double-click `run.bat` in the project folder. It will start the server and automatically launch your browser at `http://127.0.0.1:5000`.

### Option B: From PowerShell / Terminal
```powershell
cd C:\Users\asvaa\.gemini\antigravity\scratch\disease_prediction_system
python app.py
```
Open your browser and visit: `http://127.0.0.1:5000`

---

## 💡 Quick Presentation Walkthrough

1. **Start on the Home Page (`/`)**:
   - Overview of the 5 disease models and test accuracies displayed in the live engine status card.

2. **Go to "Register New Patient" (`/patient/new`)**:
   - Point out the **1-Click Demo Personas** banner.
   - Click **"Patient A: High Cardiac & Stroke Risk"** (Rajesh Sharma) or **"Patient B: Type-2 Diabetes Alert"**.
   - Show that all vitals and BMI auto-calculate immediately.
   - Click **"Run AI & Open Doctor Portal"**.

3. **In the Doctor Portal (`/doctor`)**:
   - Show the **Unified Multi-Disease Risk Index** and **Triage Status**.
   - Show the **5 Disease Cards** with calibrated probabilities.
   - Switch between disease tabs to show the **SHAP Explainability Chart** (e.g. show how Systolic BP 164 mmHg and Active Smoking drove heart risk up).
   - Click **"Print Clinical Report"** to show the professional diagnostic summary sheet.

4. **Switch to Patient Portal (`/patient`)**:
   - Show how the same patient data is translated into plain, reassuring health metrics (Health Score / 100).
   - Show the **Personalized Action Plan**: tailored dietary advice and exercise timeline.

5. **Show Architecture & Review View (`/presentation`)**:
   - Seamlessly connect the running software directly to the System Architecture and Literature Survey.
