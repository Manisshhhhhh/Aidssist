import unittest

import pandas as pd

from backend.aidssist_runtime.cache import get_cache_store
from backend.dataset_understanding import analyze_dataset, detect_domain
from backend.suggestion_engine import (
    build_suggestion_payload,
    get_user_interaction_memory,
    record_user_interaction_memory,
)


class SuggestionEngineTests(unittest.TestCase):
    def setUp(self):
        get_cache_store.cache_clear()

    def tearDown(self):
        get_cache_store.cache_clear()

    def test_analyze_dataset_detects_business_time_series_context(self):
        df = pd.DataFrame(
            {
                "order_date": pd.date_range("2025-01-01", periods=6, freq="D"),
                "revenue": [1000, 1025, 980, 1100, 1150, 1200],
                "region": ["North", "South", "East", "West", "North", "South"],
                "product": ["A", "B", "A", "C", "B", "A"],
            }
        )

        context = analyze_dataset(df)

        self.assertEqual(context["domain"], "business")
        self.assertTrue(context["is_time_series"])
        self.assertIn("revenue", context["primary_metrics"])
        self.assertIn("region", context["categorical_features"])
        self.assertIn("order_date", context["time_columns"])

    def test_detect_domain_identifies_finance_dataset(self):
        df = pd.DataFrame(
            {
                "trade_date": pd.date_range("2025-01-01", periods=4, freq="D"),
                "ticker": ["AAA", "BBB", "AAA", "CCC"],
                "return": [0.02, -0.01, 0.03, 0.01],
                "balance": [100000, 99500, 101200, 102100],
            }
        )

        self.assertEqual(detect_domain(df), "finance")

    def test_build_suggestion_payload_returns_ranked_prompts_and_recommended_step(self):
        df = pd.DataFrame(
            {
                "report_date": pd.date_range("2025-01-01", periods=8, freq="D"),
                "sales": [100, 104, 108, 115, 117, 121, 130, 135],
                "region": ["North", "South"] * 4,
            }
        )

        payload = build_suggestion_payload(df, source_fingerprint="fp-sales", recent_queries=["Show sales trend"])

        self.assertEqual(payload["domain"], "business")
        self.assertTrue(payload["context"]["is_time_series"])
        self.assertTrue(payload["suggestions"])
        self.assertEqual(payload["suggestions"][0]["rank"], 1)
        self.assertTrue(payload["recommended_next_step"])
        self.assertIn(payload["recommended_next_step"], payload["suggested_questions"])

    def test_interaction_memory_stores_queries_and_successes(self):
        record_user_interaction_memory(
            source_fingerprint="fp-memory",
            dataset_type="business",
            query="Compare sales by region",
            successful_action="Compare sales by region",
        )

        memory = get_user_interaction_memory("fp-memory")

        self.assertEqual(memory["dataset_type"], "business")
        self.assertIn("Compare sales by region", memory["queries"])
        self.assertIn("Compare sales by region", memory["successful_actions"])


if __name__ == "__main__":
    unittest.main()
