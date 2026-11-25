import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

np.random.seed(42)

def generate_clinical_dataset(n_samples=2500):
    age = np.random.normal(52, 14, n_samples).clip(18, 90)
    gender = np.random.binomial(1, 0.54, n_samples)
    height = np.random.normal(168, 9, n_samples).clip(140, 200)
    weight = np.random.normal(74, 15, n_samples).clip(45, 140)
    bmi = weight / ((height / 100) ** 2)

    smoking = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20])
    alcohol = np.random.choice([0, 1, 2], size=n_samples, p=[0.60, 0.28, 0.12])
    physical_activity = np.random.choice([0, 1, 2], size=n_samples, p=[0.35, 0.45, 0.20])
    sleep_hours = np.random.normal(6.8, 1.3, n_samples).clip(3, 11)

    systolic_bp = (105 + age * 0.4 + (bmi - 24) * 0.8 + (smoking == 2) * 12 + np.random.normal(0, 12, n_samples)).clip(90, 210)
    diastolic_bp = (68 + systolic_bp * 0.15 + np.random.normal(0, 8, n_samples)).clip(55, 125)

    fasting_glucose = (80 + age * 0.3 + (bmi - 23) * 1.5 + (physical_activity == 0) * 12 + np.random.normal(0, 20, n_samples)).clip(65, 320)
    hba1c = (4.6 + fasting_glucose * 0.024 + np.random.normal(0, 0.4, n_samples)).clip(4.2, 14.0)

    cholesterol_total = (160 + age * 0.5 + (bmi - 22) * 1.2 + (smoking > 0) * 15 + np.random.normal(0, 28, n_samples)).clip(110, 380)
    cholesterol_hdl = (55 - (bmi - 24) * 0.5 + (physical_activity == 2) * 8 + np.random.normal(0, 10, n_samples)).clip(25, 95)
    cholesterol_ldl = (cholesterol_total - cholesterol_hdl - 25).clip(50, 280)

    serum_creatinine = (0.75 + age * 0.005 + (systolic_bp > 140) * 0.3 + (fasting_glucose > 140) * 0.25 + np.random.exponential(0.3, n_samples)).clip(0.5, 6.5)
    blood_urea = (14 + serum_creatinine * 12 + np.random.normal(0, 8, n_samples)).clip(6, 110)

    liver_alt = (22 + (alcohol == 2) * 35 + (bmi > 30) * 20 + np.random.exponential(15, n_samples)).clip(10, 220)
    liver_ast = (20 + (alcohol == 2) * 40 + (liver_alt * 0.7) + np.random.exponential(12, n_samples)).clip(10, 200)
    bilirubin_total = (0.6 + (liver_alt > 60) * 0.8 + (alcohol == 2) * 0.7 + np.random.exponential(0.4, n_samples)).clip(0.2, 8.0)

    chest_pain = np.random.binomial(1, (systolic_bp > 150) * 0.35 + (smoking == 2) * 0.2 + 0.08, n_samples)
    shortness_of_breath = np.random.binomial(1, (bmi > 30) * 0.3 + (chest_pain == 1) * 0.35 + 0.1, n_samples)
    ecg_abnormal = np.random.binomial(1, (chest_pain == 1) * 0.45 + (systolic_bp > 160) * 0.3 + (age > 60) * 0.2 + 0.05, n_samples)
    retinal_scan_abnormal = np.random.binomial(1, (fasting_glucose > 140) * 0.45 + (systolic_bp > 150) * 0.35 + 0.06, n_samples)

    heart_score = (
        (age > 55) * 2.0 +
        (gender == 1) * 1.2 +
        (systolic_bp > 140) * 2.5 +
        (cholesterol_total > 220) * 2.2 +
        (cholesterol_hdl < 40) * 1.8 +
        (smoking == 2) * 2.4 +
        (chest_pain == 1) * 3.5 +
        (ecg_abnormal == 1) * 3.0 +
        np.random.normal(0, 1.0, n_samples)
    )
    y_heart = (heart_score > 7.0).astype(int)

    diabetes_score = (
        (fasting_glucose > 110) * 3.2 +
        (hba1c > 6.0) * 4.0 +
        (bmi > 28) * 2.2 +
        (age > 45) * 1.5 +
        (physical_activity == 0) * 1.6 +
        (retinal_scan_abnormal == 1) * 2.0 +
        np.random.normal(0, 1.0, n_samples)
    )
    y_diabetes = (diabetes_score > 6.5).astype(int)

    stroke_score = (
        (age > 60) * 3.0 +
        (systolic_bp > 150) * 3.5 +
        (y_heart == 1) * 2.5 +
        (fasting_glucose > 130) * 1.8 +
        (smoking == 2) * 2.2 +
        (bmi > 29) * 1.4 +
        np.random.normal(0, 1.0, n_samples)
    )
    y_stroke = (stroke_score > 7.5).astype(int)

    ckd_score = (
        (serum_creatinine > 1.4) * 4.5 +
        (blood_urea > 30) * 3.0 +
        (systolic_bp > 145) * 2.0 +
        (fasting_glucose > 140) * 2.0 +
        (age > 60) * 1.5 +
        np.random.normal(0, 1.0, n_samples)
    )
    y_ckd = (ckd_score > 6.8).astype(int)

    liver_score = (
        (liver_alt > 45) * 3.8 +
        (liver_ast > 40) * 3.2 +
        (bilirubin_total > 1.3) * 3.5 +
        (alcohol == 2) * 3.0 +
        (bmi > 30) * 1.5 +
        np.random.normal(0, 1.0, n_samples)
    )
    y_liver = (liver_score > 7.0).astype(int)

    df = pd.DataFrame({
        'age': age, 'gender': gender, 'bmi': bmi, 'smoking': smoking, 'alcohol': alcohol,
        'physical_activity': physical_activity, 'sleep_hours': sleep_hours,
        'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
        'fasting_glucose': fasting_glucose, 'hba1c': hba1c,
        'cholesterol_total': cholesterol_total, 'cholesterol_hdl': cholesterol_hdl, 'cholesterol_ldl': cholesterol_ldl,
        'serum_creatinine': serum_creatinine, 'blood_urea': blood_urea,
        'liver_alt': liver_alt, 'liver_ast': liver_ast, 'bilirubin_total': bilirubin_total,
        'chest_pain': chest_pain, 'shortness_of_breath': shortness_of_breath,
        'ecg_abnormal': ecg_abnormal, 'retinal_scan_abnormal': retinal_scan_abnormal,
        'y_heart': y_heart, 'y_diabetes': y_diabetes, 'y_stroke': y_stroke, 'y_ckd': y_ckd, 'y_liver': y_liver
    })
    return df

FEATURE_COLS = [
    'age', 'gender', 'bmi', 'smoking', 'alcohol', 'physical_activity', 'sleep_hours',
    'systolic_bp', 'diastolic_bp', 'fasting_glucose', 'hba1c',
    'cholesterol_total', 'cholesterol_hdl', 'cholesterol_ldl',
    'serum_creatinine', 'blood_urea', 'liver_alt', 'liver_ast', 'bilirubin_total',
    'chest_pain', 'shortness_of_breath', 'ecg_abnormal', 'retinal_scan_abnormal'
]

def train_and_save_models(output_path='ml_engine/trained_models.pkl'):
    print("Generating multimodal clinical dataset...")
    df = generate_clinical_dataset(3000)
    X = df[FEATURE_COLS]

    models = {}
    metrics = {}
    feature_importances = {}

    targets = {
        'heart': 'y_heart',
        'diabetes': 'y_diabetes',
        'stroke': 'y_stroke',
        'ckd': 'y_ckd',
        'liver': 'y_liver'
    }

    train_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]

    for key, target_col in targets.items():
        y_train = df[target_col].iloc[:train_idx]
        y_test = df[target_col].iloc[train_idx:]

        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)

        stack = StackingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            final_estimator=LogisticRegression(C=1.0),
            cv=3
        )
        stack.fit(X_train, y_train)

        # Train a standalone GradientBoosting for fast exact SHAP-style tree contributions
        gb_single = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
        gb_single.fit(X_train, y_train)

        test_probs = stack.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, test_probs)
        acc = accuracy_score(y_test, stack.predict(X_test))

        models[key] = stack
        metrics[key] = {'auc': float(round(auc, 4)), 'accuracy': float(round(acc, 4))}
        feature_importances[key] = {
            col: float(round(imp, 4)) for col, imp in zip(FEATURE_COLS, gb_single.feature_importances_)
        }
        print(f"[{key.upper()}] Accuracy: {acc*100:.2f}%, AUC: {auc:.4f}")

    train_meta_X = np.column_stack([models[k].predict_proba(X_train)[:, 1] for k in targets.keys()])
    y_meta_train = ((train_meta_X.max(axis=1) > 0.45) | (train_meta_X.mean(axis=1) > 0.25)).astype(int)

    meta_learner = LogisticRegression(C=1.0)
    meta_learner.fit(train_meta_X, y_meta_train)

    package = {
        'models': models,
        'meta_learner': meta_learner,
        'feature_cols': FEATURE_COLS,
        'metrics': metrics,
        'feature_importances': feature_importances,
        'baseline_means': {k: float(v) for k, v in X_train.mean().items()},
        'baseline_stds': {k: float(v) for k, v in X_train.std().items()}
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(package, output_path)
    print("Models saved successfully to", output_path)

if __name__ == '__main__':
    train_and_save_models()
