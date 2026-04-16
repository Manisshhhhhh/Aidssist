import unittest

import pandas as pd

from backend.services.data_intelligence import detect_dataset_type
from backend.services.mode_router import decide_analysis_mode


class ModeRouterTests(unittest.TestCase):
    def test_time_series_dataset_routes_to_forecast(self):
        df = pd.DataFrame(
            {
                "order_date": pd.date_range("2025-01-01", periods=24, freq="D"),
                "sales": range(100, 124),
                "region": ["North", "South"] * 12,
            }
        )

        intelligence = detect_dataset_type(df)
        decision = decide_analysis_mode(df, "predict future sales")

        self.assertTrue(intelligence["has_datetime"])
        self.assertTrue(intelligence["is_time_series"])
        self.assertEqual(decision["mode"], "forecast")
        self.assertGreater(decision["confidence"], 0.9)

    def test_student_dataset_routes_to_ml(self):
        df = pd.DataFrame(
            {
                "study_hours": [3, 4, 5, 6, 7, 8, 4, 6, 5, 7, 8, 9],
                "attendance": [75, 80, 82, 88, 90, 92, 78, 86, 84, 89, 94, 96],
                "gender": ["F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
                "final_score": [58, 62, 67, 71, 76, 81, 60, 73, 69, 78, 84, 88],
            }
        )

        intelligence = detect_dataset_type(df)
        decision = decide_analysis_mode(df, "predict final score")

        self.assertFalse(intelligence["has_datetime"])
        self.assertTrue(intelligence["is_ml_ready"])
        self.assertEqual(decision["mode"], "ml")
        self.assertGreater(decision["confidence"], 0.8)

    def test_categorical_dataset_routes_to_analysis(self):
        df = pd.DataFrame(
            {
                "region": ["North", "South", "East", "West"],
                "segment": ["SMB", "Enterprise", "SMB", "Mid-Market"],
                "status": ["Open", "Closed", "Open", "Pending"],
            }
        )

        intelligence = detect_dataset_type(df)
        decision = decide_analysis_mode(df, "show breakdown by region")

        self.assertFalse(intelligence["is_time_series"])
        self.assertFalse(intelligence["is_ml_ready"])
        self.assertEqual(decision["mode"], "analysis")


if __name__ == "__main__":
    unittest.main()
