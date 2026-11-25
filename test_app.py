import unittest
from app import app, load_patients

class TestDiseasePredictionSystem(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page(self):
        res = self.client.get('/')
        self.assertIn(b'MesoPredict', res.data)
        self.assertIn(b'Multimodal Early Disease Prediction System', res.data)
        print("[PASS] Home Page rendered successfully with MesoPredict AI branding.")

    def test_patient_entry_page(self):
        res = self.client.get('/patient/new')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Multimodal Patient Data Intake', res.data)
        self.assertIn(b'loadPreset', res.data)
        print("[PASS] Patient Intake Page rendered with 1-Click presets.")

    def test_doctor_portal(self):
        res = self.client.get('/doctor', follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Clinical Diagnostic', res.data)
        self.assertIn(b'shapChart', res.data)
        self.assertIn(b'Print Clinical Report', res.data)
        print("[PASS] Doctor Portal rendered with SHAP chart & clinical report.")

    def test_patient_portal(self):
        res = self.client.get('/patient', follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Personal Health', res.data)
        self.assertIn(b'Unified Health Score', res.data)
        self.assertIn(b'Dietary Guidance', res.data)
        print("[PASS] Patient Portal rendered with health index and lifestyle tips.")

    def test_presentation_view(self):
        res = self.client.get('/presentation')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'System Design Architecture', res.data)
        self.assertIn(b'Literature Review', res.data)
        print("[PASS] Architecture & Literature Review companion rendered.")

    def test_api_predict(self):
        payload = {
            'age': 55, 'gender': 1, 'height': 175, 'weight': 80, 'bmi': 26.1,
            'smoking': 2, 'alcohol': 1, 'physical_activity': 0, 'sleep_hours': 6.0,
            'systolic_bp': 160, 'diastolic_bp': 95, 'fasting_glucose': 140, 'hba1c': 6.6,
            'cholesterol_total': 240, 'cholesterol_hdl': 38, 'cholesterol_ldl': 160,
            'serum_creatinine': 1.2, 'blood_urea': 20, 'liver_alt': 35, 'liver_ast': 30, 'bilirubin_total': 0.8,
            'chest_pain': 1, 'shortness_of_breath': 1, 'ecg_abnormal': 1, 'retinal_scan_abnormal': 0
        }
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('diseases', data)
        self.assertIn('unified_risk_score', data)
        self.assertIn('overall_health_score', data)
        self.assertGreater(data['diseases']['heart']['percentage'], 70.0)
        print(f"[PASS] /api/predict calculated Heart Risk: {data['diseases']['heart']['percentage']}%")

    def test_patient_save_flow(self):
        form_data = {
            'name': 'Test New Patient',
            'age': '48',
            'gender': '1',
            'height': '172',
            'weight': '78',
            'blood_group': 'B+',
            'smoking': '1',
            'alcohol': '0',
            'physical_activity': '1',
            'sleep_hours': '7.0',
            'systolic_bp': '135',
            'diastolic_bp': '85',
            'fasting_glucose': '110',
            'hba1c': '5.8',
            'cholesterol_total': '205',
            'cholesterol_hdl': '46',
            'cholesterol_ldl': '130',
            'serum_creatinine': '1.05',
            'blood_urea': '17',
            'liver_alt': '30',
            'liver_ast': '28',
            'bilirubin_total': '0.7',
            'notes': 'Test patient automated verification',
            'target': 'doctor'
        }
        res = self.client.post('/api/patient/save', data=form_data, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Test New Patient', res.data)
        print("[PASS] New Patient saved and immediately reflected in Doctor Portal.")

if __name__ == '__main__':
    unittest.main()
