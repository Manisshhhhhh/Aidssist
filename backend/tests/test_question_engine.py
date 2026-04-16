import unittest

import pandas as pd

from backend.question_engine import build_question_payload, detect_domain, generate_suggested_questions


class QuestionEngineTests(unittest.TestCase):
    def test_detect_domain_identifies_business_dataset(self):
        df = pd.DataFrame(
            {
                "order_date": pd.date_range("2025-01-01", periods=5, freq="D"),
                "sales": [100, 120, 140, 160, 180],
                "product": ["A", "B", "A", "C", "B"],
            }
        )

        self.assertEqual(detect_domain(df), "business")

    def test_generate_suggested_questions_returns_domain_questions(self):
        df = pd.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "blood_pressure": [120, 145, 160],
                "risk_score": [0.2, 0.7, 0.9],
            }
        )

        questions = generate_suggested_questions(df, limit=4)

        self.assertTrue(questions)
        self.assertTrue(any("abnormal" in question.lower() or "risk" in question.lower() for question in questions))

    def test_build_question_payload_returns_context_and_recommended_next_step(self):
        df = pd.DataFrame(
            {
                "order_date": pd.date_range("2025-01-01", periods=5, freq="D"),
                "sales": [100, 120, 140, 160, 180],
                "region": ["North", "South", "East", "West", "North"],
            }
        )

        payload = build_question_payload(df, source_fingerprint="fp-questions", recent_queries=["Show sales trend"])

        self.assertEqual(payload["domain"], "business")
        self.assertIn("context", payload)
        self.assertIn("suggestions", payload)
        self.assertTrue(payload["suggestions"])
        self.assertTrue(payload["recommended_next_step"])
        self.assertTrue(payload["suggested_questions"])


if __name__ == "__main__":
    unittest.main()
