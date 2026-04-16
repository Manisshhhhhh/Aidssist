import unittest

import pandas as pd

from backend.services.ml_intelligence import build_ml_intelligence
from backend.services.ml_postprocessor import postprocess_ml_output
from backend.services.ml_schema_validator import validate_ml_output


def _collect_none_paths(value, prefix="root"):
    paths: list[str] = []
    if value is None:
        return [prefix]
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(_collect_none_paths(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_collect_none_paths(item, f"{prefix}[{index}]"))
    return paths


class MLOutputStructureTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "study_hours_per_day": [3.0, 3.5, 4.0, 4.2, 4.8, 5.0, 5.5, 6.0, 6.2, 6.8, 7.0, 7.5],
                "exam_pressure": [9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4],
                "sleep_hours": [5.5, 5.8, 6.0, 6.2, 6.5, 6.7, 7.0, 7.2, 7.3, 7.5, 7.7, 8.0],
                "stress_level": [8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3],
            }
        )

    def test_ml_output_structure_is_strict_complete_and_non_null(self):
        raw_output = build_ml_intelligence(
            self.df,
            user_query="Predict study_hours_per_day",
            target_hint="study_hours_per_day",
            insights=["Exam pressure and sleep habits influence study time."],
        )
        ml_output = postprocess_ml_output(raw_output, self.df)

        validate_ml_output(ml_output)

        required_keys = {
            "target",
            "problem_type",
            "features",
            "metrics",
            "top_features",
            "feature_importance",
            "predictions_sample",
            "data_quality_score",
            "confidence",
            "warnings",
            "recommendations",
        }
        self.assertEqual(set(ml_output.keys()), required_keys)
        self.assertFalse(_collect_none_paths(ml_output))
        self.assertTrue(ml_output["features"])
        self.assertIsInstance(ml_output["metrics"]["mae"], float)
        self.assertIsInstance(ml_output["metrics"]["r2"], float)
        self.assertGreaterEqual(ml_output["metrics"]["mae"], 0.0)
        self.assertGreaterEqual(ml_output["metrics"]["r2"], 0.0)
        self.assertLessEqual(ml_output["data_quality_score"], 1.0)
        self.assertLessEqual(ml_output["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
