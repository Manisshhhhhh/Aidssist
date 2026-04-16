import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from sklearn.linear_model import LinearRegression

from backend.services.failure_logging import failure_log, get_failure_patterns, log_failure
from backend.services.limitations import build_limitations
from backend.services.model_quality import (
    build_explanation,
    build_simple_prediction_diagnostics,
    evaluate_model,
    evaluate_model_with_warnings,
    interpret_model_quality,
)
from backend.services.result_consistency import (
    build_analysis_consistency,
    build_reproducibility_metadata,
    hash_result,
    normalize_query,
)
from backend.services.trust_layer import assess_risk
from backend.workflow_store import WorkflowStore


class ReliabilityServicesTests(unittest.TestCase):
    def setUp(self):
        failure_log.clear()

    def test_evaluate_model_handles_valid_and_invalid_inputs(self):
        metrics = evaluate_model([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(metrics["mae"], 0.3333333333, places=4)
        self.assertIsNotNone(metrics["r2"])

        invalid_metrics, warnings = evaluate_model_with_warnings([1, None, "bad"], [1, 2, 3])
        self.assertIsNotNone(invalid_metrics["mae"])
        self.assertTrue(warnings)

        empty_metrics, empty_warnings = evaluate_model_with_warnings([], [])
        self.assertIsNone(empty_metrics["mae"])
        self.assertTrue(empty_warnings)

    def test_build_explanation_supports_linear_and_tree_models(self):
        linear_model = LinearRegression()
        linear_model.coef_ = [0.2, -0.8]  # type: ignore[attr-defined]
        explanation = build_explanation(model=linear_model, feature_names=["price", "discount"])
        self.assertEqual(explanation["top_features"][0], "discount")

        class TreeStub:
            feature_importances_ = [0.7, 0.3]

        tree_model = TreeStub()
        tree_explanation = build_explanation(model=tree_model, feature_names=["region", "channel"])
        self.assertEqual(tree_explanation["top_features"][0], "region")

    def test_build_simple_prediction_diagnostics_returns_metrics_and_explanation(self):
        df = pd.DataFrame(
            {
                "order_date": pd.date_range("2025-01-01", periods=12, freq="D"),
                "sales": [100, 102, 105, 107, 110, 112, 115, 117, 119, 121, 124, 126],
            }
        )
        diagnostics = build_simple_prediction_diagnostics(
            df,
            target_column="sales",
            datetime_column="order_date",
        )
        self.assertIsNotNone(diagnostics["model_metrics"]["mae"])
        self.assertTrue(diagnostics["explanation"]["top_features"])
        self.assertIn(diagnostics["model_quality"], {"strong", "moderate", "weak"})

    def test_model_quality_risk_and_reproducibility_metadata_are_derived(self):
        self.assertEqual(interpret_model_quality(1.0, 0.9), "strong")
        self.assertEqual(interpret_model_quality(1.0, 0.6), "moderate")
        self.assertEqual(interpret_model_quality(1.0, 0.1), "weak")
        self.assertEqual(assess_risk({"score": 9.0}, "strong"), "low")
        self.assertEqual(assess_risk({"score": 4.0}, "moderate"), "high")

        metadata = build_reproducibility_metadata(
            source_fingerprint="fingerprint-1",
            pipeline_trace=[{"stage": "user_query", "status": "completed"}],
            result_hash="hash-1",
            consistency_payload={"prior_hash_count": 2, "inconsistency_detected": False, "consistency_validated": True},
        )
        self.assertEqual(metadata["dataset_fingerprint"], "fingerprint-1")
        self.assertEqual(metadata["result_hash"], "hash-1")
        self.assertTrue(metadata["pipeline_trace_hash"])

    def test_build_limitations_uses_rules_and_llm_fallback_logging(self):
        df = pd.DataFrame({"sales": [100, None, None, None, None]})
        with mock.patch("backend.prompt_pipeline.analyze_system_weaknesses", return_value="- Extra limitation"):
            limitations = build_limitations(
                query="Predict sales",
                result={"status": "ok"},
                df=df,
                warnings=["Data missing"],
                data_score={"score": 55},
                data_quality={"score": 4.5},
                model_metrics={"mae": 12.0, "r2": 0.2},
                model_quality="weak",
                risk="high due to poor data quality",
                explanation={"top_features": [], "impact": []},
                inconsistency_detected=True,
                analysis_type="time_series",
                use_llm=True,
            )
        self.assertTrue(any("Extra limitation" in item for item in limitations))
        self.assertTrue(any("Prediction confidence is low" in item for item in limitations))

        with mock.patch("backend.prompt_pipeline.analyze_system_weaknesses", side_effect=RuntimeError("boom")):
            fallback_limitations = build_limitations(
                query="Predict sales",
                result={"status": "ok"},
                df=df,
                warnings=["Data missing"],
                data_score={"score": 55},
                data_quality={"score": 4.5},
                model_metrics={"mae": None, "r2": None},
                model_quality="weak",
                risk="high due to poor data quality",
                explanation={"top_features": [], "impact": []},
                inconsistency_detected=False,
                analysis_type="time_series",
                use_llm=True,
            )
        self.assertTrue(fallback_limitations)
        self.assertTrue(failure_log)

    def test_result_consistency_detects_prior_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = WorkflowStore(Path(temp_dir) / "consistency.sqlite3")
            run_record = store.build_run_record(
                workflow_id="wf-1",
                workflow_version=1,
                workflow_name="Ops",
                source_fingerprint="fingerprint-1",
                source_label="sales.csv",
                validation_findings=[],
                cleaning_actions=[],
                generated_code="result = 1",
                final_status="PASSED",
                error_message=None,
                export_artifacts=["json"],
                analysis_query="predict sales",
                result_summary="ok",
                result_hash=hash_result({"value": 1}),
            )
            store.record_run(run_record)
            consistency = build_analysis_consistency(
                store=store,
                result={"value": 2},
                source_fingerprint="fingerprint-1",
                query="Predict   Sales",
                analysis_intent="prediction",
            )
            store.close()

        self.assertTrue(consistency["inconsistency_detected"])
        self.assertEqual(normalize_query("Predict   Sales"), "predict sales")

    def test_log_failure_persists_to_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = WorkflowStore(Path(temp_dir) / "failures.sqlite3")
            log_failure("query", RuntimeError("broken"), "execution", store=store, metadata={"job_id": "1"})
            failure_records = store.list_failure_logs()
            store.close()

        self.assertEqual(len(failure_records), 1)
        self.assertEqual(failure_records[0].stage, "execution")

    def test_log_failure_updates_pattern_counters(self):
        log_failure("forecast next month sales", RuntimeError("missing date column"), "forecast_preparation")
        patterns = get_failure_patterns()
        self.assertGreaterEqual(patterns["missing_date_column"], 1)


if __name__ == "__main__":
    unittest.main()
