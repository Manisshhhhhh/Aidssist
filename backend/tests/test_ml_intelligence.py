import unittest

import pandas as pd

from backend.analysis_contract import build_analysis_contract
from backend.services.explainer import explain_model
from backend.services.feature_selector import select_features
from backend.services.ml_intelligence import build_ml_intelligence
from backend.services.ml_postprocessor import postprocess_ml_output
from backend.services.ml_schema_validator import validate_ml_output
from backend.services.model_trainer import train_model
from backend.services.target_detector import detect_target_column
from backend.suggestion_engine import build_suggestion_payload


class MLIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "study_hours": [2, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7],
                "attendance": [68, 70, 74, 77, 81, 84, 88, 91, 73, 76, 82, 86],
                "exam_pressure": [9, 8, 8, 7, 7, 6, 5, 4, 8, 7, 6, 5],
                "student_id": list(range(100, 112)),
                "final_score": [52, 57, 61, 66, 71, 76, 82, 88, 63, 69, 75, 81],
            }
        )

    def test_detect_target_column_prefers_named_business_outcome(self):
        detection = detect_target_column(self.df, "Predict final_score")

        self.assertEqual(detection["target"], "final_score")
        self.assertEqual(detection["type"], "regression")
        self.assertGreater(detection["confidence"], 0.9)

    def test_feature_selector_ranks_signal_features_and_skips_ids(self):
        payload = select_features(self.df, "final_score")

        self.assertIn("study_hours", payload["selected_features"])
        self.assertTrue(any(feature in payload["selected_features"] for feature in ("attendance", "exam_pressure")))
        self.assertNotIn("student_id", payload["selected_features"])
        self.assertTrue(payload["importance_scores"])

    def test_model_trainer_and_explainer_produce_metrics_and_top_features(self):
        selected = select_features(self.df, "final_score")
        training = train_model(self.df, "final_score", selected["selected_features"])
        explanation = explain_model(training["model"], selected["selected_features"])

        self.assertIsNotNone(training["metrics"]["mae"])
        self.assertIn(training["model_name"], {"linear_regression", "random_forest"})
        self.assertTrue(explanation["top_features"])

    def test_build_ml_intelligence_returns_contract_ready_payload(self):
        raw_payload = build_ml_intelligence(
            self.df,
            user_query="What affects final_score?",
            target_hint="final_score",
            insights=["Study hours lead to better outcomes."],
        )
        payload = postprocess_ml_output(raw_payload, self.df)
        validate_ml_output(payload)

        self.assertEqual(payload["target"], "final_score")
        self.assertEqual(payload["problem_type"], "regression")
        self.assertTrue(payload["features"])
        self.assertTrue(payload["recommendations"])
        self.assertTrue(payload["top_features"])
        self.assertTrue(payload["predictions_sample"])
        self.assertIn("study_hours", payload["feature_importance"])

    def test_build_suggestion_payload_includes_target_driven_questions(self):
        payload = build_suggestion_payload(self.df, limit=6)

        self.assertIn("What affects final_score?", payload["suggested_questions"])
        self.assertIn("Predict final_score", payload["suggested_questions"])
        self.assertIn("Improve final_score", payload["suggested_questions"])

    def test_analysis_contract_preserves_ml_intelligence_and_seeds_recommendations(self):
        ml_intelligence = {
            "target": "final_score",
            "problem_type": "regression",
            "features": ["study_hours", "attendance"],
            "metrics": {"mae": 1.8, "r2": 0.91},
            "top_features": ["study_hours", "attendance"],
            "feature_importance": {"study_hours": 0.57, "attendance": 0.43},
            "predictions_sample": [52.0, 57.0, 61.0, 66.0, 71.0],
            "data_quality_score": 0.9,
            "confidence": 0.9,
            "warnings": [],
            "recommendations": ["Increase study_hours support to improve final_score."],
        }

        contract = build_analysis_contract(
            query="Predict final_score",
            df=self.df,
            result={"ok": True},
            executed_code="result = {'ok': True}",
            plan={"intent": "prediction", "tool_used": "PYTHON", "analysis_mode": "prediction"},
            preflight={"warnings": [], "blocking_errors": []},
            method="deterministic_ml",
            model_metrics={"mae": 1.8, "r2": 0.91},
            explanation={"top_features": ["study_hours"], "impact": [0.7]},
            ml_intelligence=ml_intelligence,
        )

        self.assertEqual(contract["ml_intelligence"]["target"], "final_score")
        self.assertIn("Increase study_hours support to improve final_score.", contract["recommendations"])


if __name__ == "__main__":
    unittest.main()
