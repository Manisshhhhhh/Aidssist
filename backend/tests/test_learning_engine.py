import os
import tempfile
import unittest

import pandas as pd

from backend.aidssist_runtime.cache import get_cache_store
from backend.services.decision_engine import build_decision_layer
from backend.services.learning_engine import (
    adjust_confidence,
    adjust_risk,
    get_learning_patterns,
    learn_from_outcomes,
    refresh_learning_patterns,
)
from backend.workflow_store import WorkflowStore


class LearningEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "learning.sqlite3")
        get_cache_store.cache_clear()

    def tearDown(self):
        get_cache_store.cache_clear()
        self.temp_dir.cleanup()

    def test_learn_from_outcomes_aggregates_success_rates_and_impact(self):
        history = [
            {
                "decision_json": {"action": "Increase focus on North because it currently leads sales."},
                "outcome": "Revenue increased by 12% and the region outperformed expectations.",
            },
            {
                "decision_json": {"action": "Increase focus on South because it currently leads sales."},
                "outcome": "Revenue increased 8% after the region expansion.",
            },
            {
                "decision_json": {"action": "Increase focus on East because it currently leads sales."},
                "outcome": "The move failed and revenue declined by 4%.",
            },
        ]

        patterns = learn_from_outcomes(history)

        self.assertIn("expand_region", patterns)
        self.assertAlmostEqual(patterns["expand_region"]["success_rate"], 2 / 3, places=2)
        self.assertAlmostEqual(patterns["expand_region"]["avg_impact"], (12 + 8 - 4) / 3, places=2)
        self.assertEqual(patterns["expand_region"]["sample_size"], 3)

    def test_adjust_confidence_and_risk_use_history_and_small_samples(self):
        strong_pattern = {
            "expand_region": {"success_rate": 0.8, "avg_impact": 10.2, "sample_size": 5, "uncertainty": "low"}
        }
        weak_pattern = {
            "expand_region": {"success_rate": 0.2, "avg_impact": -6.0, "sample_size": 5, "uncertainty": "low"}
        }
        tiny_pattern = {
            "expand_region": {"success_rate": 1.0, "avg_impact": 12.0, "sample_size": 2, "uncertainty": "high"}
        }

        self.assertEqual(adjust_confidence("medium", strong_pattern, "expand_region"), "high")
        self.assertEqual(adjust_risk("medium", strong_pattern, "expand_region"), "low")
        self.assertEqual(adjust_confidence("high", weak_pattern, "expand_region"), "medium")
        self.assertEqual(adjust_risk("medium", weak_pattern, "expand_region"), "high")
        self.assertEqual(adjust_confidence("high", tiny_pattern, "expand_region"), "low")

    def test_get_learning_patterns_refreshes_cached_outcomes(self):
        store = WorkflowStore(self.db_path)
        try:
            record = store.build_decision_history_record(
                job_id="job-1",
                forecast_artifact_id=None,
                source_fingerprint="fp-learning",
                query="compare regions",
                decision={
                    "decision_id": "decision-1",
                    "action": "Increase focus on North because it currently leads sales.",
                    "expected_impact": "Increase revenue by ~10%",
                    "confidence": "medium",
                    "risk_level": "medium",
                    "priority": "HIGH",
                    "reasoning": "North leads.",
                },
                decision_confidence="medium",
                result_hash="hash-1",
            )
            store.record_decision_history(record)

            initial_patterns = get_learning_patterns(store, "fp-learning", refresh=True)
            self.assertEqual(initial_patterns, {})

            store.update_decision_outcome(
                record.decision_history_id,
                "Revenue increased by 11% after the region expansion.",
            )
            refreshed_patterns = refresh_learning_patterns(store, "fp-learning")
            cached_patterns = get_learning_patterns(store, "fp-learning")
        finally:
            store.close()

        self.assertIn("expand_region", refreshed_patterns)
        self.assertEqual(refreshed_patterns, cached_patterns)
        self.assertEqual(refreshed_patterns["expand_region"]["sample_size"], 1)

    def test_build_decision_layer_applies_learning_adjustments(self):
        result = pd.DataFrame(
            {
                "region": ["North", "South"],
                "sales": [120.0, 95.0],
            }
        )
        learning_patterns = {
            "expand_region": {
                "success_rate": 0.8,
                "avg_impact": 10.2,
                "sample_size": 5,
                "uncertainty": "low",
            }
        }

        layer = build_decision_layer(
            result,
            ["North leads current sales output."],
            plan={"intent": "comparison", "analysis_type": "aggregation", "group_column": "region", "metric_column": "sales"},
            model_quality="moderate",
            data_quality={
                "score": 7.5,
                "issues": [],
                "profile": {"missing_percent": {"sales": 0.0}},
            },
            reproducibility={"consistent_with_prior_runs": True, "consistency_validated": True},
            risk="medium due to moderate model quality",
            warnings=[],
            learning_patterns=learning_patterns,
        )

        self.assertTrue(layer["decisions"])
        self.assertEqual(layer["top_decision"]["confidence"], "high")
        self.assertEqual(layer["top_decision"]["risk_level"], "low")
        self.assertEqual(layer["top_decision"]["decision_performance"]["sample_size"], 5)
        self.assertTrue(layer["learning_insights"]["patterns"])
        self.assertIn("confidence", layer["learning_insights"]["confidence_adjustment"].lower())


if __name__ == "__main__":
    unittest.main()
