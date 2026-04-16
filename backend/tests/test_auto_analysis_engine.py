from __future__ import annotations

import unittest

import pandas as pd

from backend.services.agent_engine import analysis_memory
from backend.services.auto_analysis_engine import build_auto_analysis_payload, plan_analysis


class AutoAnalysisEngineTests(unittest.TestCase):
    def setUp(self):
        analysis_memory.clear()

    def _print_payload(self, label: str, payload: dict) -> None:
        print(f"\nAUTO ANALYSIS DATASET: {label}")
        print(payload)

    def test_business_dataset_generates_concise_non_redundant_tasks(self):
        df = pd.DataFrame(
            {
                "order_id": [1001, 1002, 1003, 1004, 1005],
                "region": ["North", "South", "North", "East", None],
                "category": ["Software", "Hardware", "Software", "Services", "Hardware"],
                "sales": [1200, 900, 1450, 700, 980],
                "profit": [300, 180, 350, 140, 200],
            }
        )

        tasks = plan_analysis(df)
        payload = build_auto_analysis_payload(df)
        self._print_payload("business", payload)

        self.assertLessEqual(len(tasks), 5)
        self.assertEqual(len(tasks), len(set(tasks)))
        self.assertIn("Show dataset summary", tasks)
        self.assertIn("Check missing values", tasks)
        self.assertTrue(any(task.startswith("Find top categories") for task in tasks))
        self.assertIn("Compute averages", tasks)
        self.assertFalse(any("trend over time" in task.lower() for task in tasks))
        self.assertTrue(payload["auto_analysis"]["summary"])

    def test_medical_dataset_highlights_missing_values_and_categories(self):
        df = pd.DataFrame(
            {
                "patient_id": ["P-1", "P-2", "P-3", "P-4", "P-5"],
                "country": ["India", "India", "US", "US", "Brazil"],
                "diagnosis": ["Flu", None, "COVID", "COVID", "Flu"],
                "severity": ["Low", "Medium", "High", "High", "Low"],
                "age": [29, 44, 61, 57, 33],
                "deaths": [0, 0, 1, 1, 0],
            }
        )

        payload = build_auto_analysis_payload(df)
        self._print_payload("medical", payload)

        results = payload["auto_analysis"]["results"]
        self.assertLessEqual(len(results), 5)
        self.assertTrue(any(result["task"] == "Check missing values" for result in results))
        self.assertTrue(any("missing" in result["insight"].lower() for result in results))
        self.assertTrue(any(task.startswith("Find top categories") for task in payload["auto_analysis"]["tasks"]))
        self.assertEqual(len(payload["auto_analysis"]["summary"]), len(set(payload["auto_analysis"]["summary"])))

    def test_time_series_dataset_adds_trend_analysis(self):
        df = pd.DataFrame(
            {
                "report_date": pd.date_range("2025-01-01", periods=6, freq="D"),
                "region": ["North", "North", "South", "South", "West", "West"],
                "confirmed_cases": [10, 12, 20, 18, 7, 9],
                "deaths": [1, 1, 2, 2, 0, 1],
            }
        )

        payload = build_auto_analysis_payload(df)
        self._print_payload("time_series", payload)

        tasks = payload["auto_analysis"]["tasks"]
        self.assertTrue(any("trend over time" in task.lower() for task in tasks))
        trend_result = next(
            result
            for result in payload["auto_analysis"]["results"]
            if "trend over time" in result["task"].lower()
        )
        self.assertEqual(trend_result["result"]["type"], "table")
        self.assertIn("report_date", trend_result["result"]["columns"])
        self.assertTrue(payload["auto_analysis"]["summary"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
