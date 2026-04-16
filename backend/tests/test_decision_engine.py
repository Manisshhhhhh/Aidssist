import unittest

import pandas as pd

from backend.services.decision_engine import (
    build_decision_layer,
    compute_decision_confidence,
    compute_decision_risk,
    estimate_expected_impact,
    rank_decisions,
)


class DecisionEngineTests(unittest.TestCase):
    def test_rank_decisions_prefers_high_impact_high_confidence_low_risk(self):
        ranked = rank_decisions(
            [
                {
                    "decision_id": "a",
                    "action": "Expand the strongest segment",
                    "expected_impact": "Increase revenue by ~12%",
                    "confidence": "high",
                    "risk_level": "low",
                    "reasoning": "Strong leader.",
                },
                {
                    "decision_id": "b",
                    "action": "Investigate a weak segment",
                    "expected_impact": "Improve performance by ~4%",
                    "confidence": "medium",
                    "risk_level": "medium",
                    "reasoning": "Mixed signals.",
                },
            ]
        )

        self.assertEqual(ranked[0]["decision_id"], "a")
        self.assertEqual(ranked[0]["priority"], "HIGH")
        self.assertIn(ranked[1]["priority"], {"MEDIUM", "LOW"})

    def test_rank_decisions_pushes_weak_high_risk_actions_down(self):
        ranked = rank_decisions(
            [
                {
                    "decision_id": "strong",
                    "action": "Scale the leading region",
                    "expected_impact": "Increase revenue by ~8%",
                    "confidence": "medium",
                    "risk_level": "low",
                    "reasoning": "Reliable enough.",
                },
                {
                    "decision_id": "weak",
                    "action": "Commit to a fragile signal",
                    "expected_impact": "Increase revenue by ~10%",
                    "confidence": "low",
                    "risk_level": "high",
                    "reasoning": "Unreliable.",
                },
            ]
        )

        self.assertEqual(ranked[0]["decision_id"], "strong")
        self.assertEqual(ranked[-1]["decision_id"], "weak")

    def test_compute_decision_confidence_uses_trust_signals(self):
        self.assertEqual(
            compute_decision_confidence(
                model_quality="strong",
                data_quality_score=9.0,
                consistency_validated=True,
                inconsistency_detected=False,
            ),
            "high",
        )
        self.assertEqual(
            compute_decision_confidence(
                model_quality="moderate",
                data_quality_score=6.5,
                consistency_validated=False,
                inconsistency_detected=False,
            ),
            "medium",
        )
        self.assertEqual(
            compute_decision_confidence(
                model_quality="weak",
                data_quality_score=8.0,
                consistency_validated=True,
                inconsistency_detected=False,
            ),
            "low",
        )
        self.assertEqual(
            compute_decision_confidence(
                model_quality="strong",
                data_quality_score=9.0,
                consistency_validated=True,
                inconsistency_detected=True,
            ),
            "low",
        )

    def test_compute_decision_risk_combines_global_risk_and_missingness(self):
        self.assertEqual(
            compute_decision_risk(
                data_quality_score=9.0,
                model_quality="strong",
                missing_percent=5.0,
                global_risk="low",
            ),
            "low",
        )
        self.assertEqual(
            compute_decision_risk(
                data_quality_score=4.5,
                model_quality="moderate",
                missing_percent=8.0,
                global_risk="medium",
            ),
            "high",
        )
        self.assertEqual(
            compute_decision_risk(
                data_quality_score=7.0,
                model_quality="moderate",
                missing_percent=18.0,
                global_risk="medium",
            ),
            "medium",
        )

    def test_estimate_expected_impact_handles_forecast_aggregation_and_fallback(self):
        forecast_decision = {
            "action": "Lean into demand growth",
            "_impact_context": {
                "prediction": {"metric_name": "revenue", "delta_ratio": 0.1},
                "aggregation": {},
            },
            "_data_quality_score": 8.5,
        }
        aggregation_decision = {
            "action": "Rebalance lagging regions",
            "_impact_context": {
                "prediction": {},
                "aggregation": {
                    "group_column": "region",
                    "metric_column": "sales",
                    "spread_ratio": 0.08,
                },
            },
            "_data_quality_score": 8.5,
        }
        fallback_decision = {
            "action": "Improve data quality before rollout",
            "_impact_context": {"prediction": {}, "aggregation": {}},
            "_data_quality_score": 4.0,
        }

        self.assertIn(
            "~10%",
            estimate_expected_impact(
                forecast_decision,
                result=None,
                plan={"analysis_type": "time_series", "intent": "prediction"},
                insights=[],
            ),
        )
        self.assertIn(
            "~8%",
            estimate_expected_impact(
                aggregation_decision,
                result=None,
                plan={"analysis_type": "aggregation", "intent": "comparison"},
                insights=[],
            ),
        )
        self.assertEqual(
            estimate_expected_impact(
                fallback_decision,
                result=None,
                plan={"analysis_type": "general", "intent": "analysis"},
                insights=[],
            ),
            "Improve data reliability for the next decision cycle",
        )

    def test_build_decision_layer_gracefully_degrades_for_empty_outputs(self):
        layer = build_decision_layer(
            None,
            [],
            plan={"intent": "analysis", "analysis_type": "general"},
            model_quality="weak",
            data_quality={"score": 4.0, "issues": [], "profile": {}},
            reproducibility={"consistent_with_prior_runs": True, "consistency_validated": False},
            risk="high due to poor data quality",
            warnings=["No usable output"],
        )

        self.assertEqual(layer["decisions"], [])
        self.assertIsNone(layer["top_decision"])
        self.assertEqual(layer["decision_confidence"], "low")

    def test_build_decision_layer_generates_ranked_structured_decisions(self):
        result = pd.DataFrame(
            {
                "region": ["North", "South"],
                "sales": [120.0, 100.0],
            }
        )

        layer = build_decision_layer(
            result,
            ["North leads visible sales output."],
            plan={"intent": "comparison", "analysis_type": "aggregation", "group_column": "region", "metric_column": "sales"},
            model_quality="moderate",
            data_quality={
                "score": 7.8,
                "issues": [],
                "profile": {"missing_percent": {"sales": 0.0}},
            },
            reproducibility={"consistent_with_prior_runs": True, "consistency_validated": True},
            risk="medium due to moderate model quality",
            warnings=[],
        )

        self.assertTrue(layer["decisions"])
        self.assertIn("action", layer["top_decision"])
        self.assertIn(layer["decision_confidence"], {"high", "medium", "low"})
        self.assertTrue(layer["risk_summary"])


if __name__ == "__main__":
    unittest.main()
