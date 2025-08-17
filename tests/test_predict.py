import unittest
from src.predict import predict_disease

class TestPrediction(unittest.TestCase):
    def test_known_symptoms(self):
        result = predict_disease("skin_rash,fever,headache")
        self.assertIn("Final Prediction", result)
        self.assertTrue(isinstance(result["Final Prediction"], str))

    def test_unknown_symptom(self):
        result = predict_disease("unknown_symptom")
        self.assertIn("Final Prediction", result)

if __name__ == "__main__":
    unittest.main()
