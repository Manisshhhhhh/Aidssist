import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from backend.aidssist_runtime.config import get_settings
from backend.aidssist_runtime.storage import get_object_store
from backend.forecasting import (
    ForecastConfig,
    HORIZON_LABELS,
    auto_detect_kpi,
    auto_detect_time_column,
    build_auto_forecast_config,
    build_forecast_eligibility,
    detect_date_column,
    is_time_series_dataset,
    persist_forecast_artifact,
    prepare_time_series,
    run_forecast_pipeline,
    suggest_forecast_mapping,
    validate_time_series,
    validate_forecast_config,
)


class ForecastingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app_state_dir = Path(self.temp_dir.name) / ".aidssist"
        self.config_patch = mock.patch("backend.aidssist_runtime.config.APP_STATE_DIR", self.app_state_dir)
        self.config_patch.start()
        get_settings.cache_clear()
        get_object_store.cache_clear()

        dates = pd.date_range("2025-01-01", periods=48, freq="D")
        self.df = pd.DataFrame(
            {
                "order_date": dates,
                "sales": np.linspace(100.0, 220.0, len(dates)),
                "discount_rate": np.tile([0.05, 0.08, 0.06], 16)[: len(dates)],
                "region": np.tile(["North", "South"], 24),
            }
        )

    def tearDown(self):
        get_object_store.cache_clear()
        get_settings.cache_clear()
        self.config_patch.stop()
        self.temp_dir.cleanup()

    def test_suggest_forecast_mapping_prefers_date_and_sales_columns(self):
        suggestions = suggest_forecast_mapping(self.df)

        self.assertEqual(suggestions["date_column"], "order_date")
        self.assertEqual(suggestions["target_column"], "sales")
        self.assertNotIn("order_date", suggestions["target_candidates"])
        self.assertIn("discount_rate", suggestions["driver_columns"])
        self.assertEqual(auto_detect_time_column(self.df), "order_date")
        self.assertEqual(auto_detect_kpi(self.df), "sales")

    def test_build_auto_forecast_config_returns_preview_payload(self):
        auto_config = build_auto_forecast_config(self.df)

        self.assertEqual(auto_config["date_column"], "order_date")
        self.assertEqual(auto_config["target"], "sales")
        self.assertTrue(auto_config["forecast_allowed"])
        self.assertIn(auto_config["confidence"], {"high", "medium"})
        self.assertIn(auto_config["horizon"], {"next_week", "next_month", "next_quarter", "next_year"})
        self.assertGreater(auto_config["data_points"], 0)

    def test_is_time_series_dataset_allows_valid_series(self):
        allowed, detected_column = is_time_series_dataset(self.df)

        self.assertTrue(allowed)
        self.assertEqual(detected_column, "order_date")

    def test_build_forecast_eligibility_blocks_non_time_dataset(self):
        non_time_df = pd.DataFrame(
            {
                "sales": [100, 120, 140, 160],
                "profit": [25, 30, 35, 40],
                "region": ["North", "South", "East", "West"],
            }
        )

        eligibility = build_forecast_eligibility(non_time_df)

        self.assertFalse(eligibility["allowed"])
        self.assertEqual(eligibility["reason"], "No valid time column detected")
        self.assertIsNone(eligibility["detected_time_column"])
        self.assertIn("Use analysis mode instead", eligibility["suggestions"])

    def test_detect_date_column_prefers_real_date_field_in_mixed_dataset(self):
        mixed_df = pd.DataFrame(
            {
                "year": [2024, 2024, 2024, 2024, 2024],
                "event_date": ["2025-01-01", "2025-01-02", None, "2025-01-04", "2025-01-05"],
                "sales": [100, 120, 118, 125, 130],
            }
        )

        allowed, detected_column = is_time_series_dataset(mixed_df)

        self.assertEqual(detect_date_column(mixed_df), "event_date")
        self.assertTrue(allowed)
        self.assertEqual(detected_column, "event_date")

    def test_validate_forecast_config_blocks_incompatible_horizon(self):
        monthly_df = self.df.copy()
        monthly_df["order_date"] = pd.date_range("2024-01-31", periods=len(monthly_df), freq="ME")
        config = ForecastConfig(
            date_column="order_date",
            target_column="sales",
            aggregation_frequency="M",
            horizon="next_week",
        )

        validation = validate_forecast_config(monthly_df, config)

        self.assertFalse(validation.is_valid)
        self.assertIn(HORIZON_LABELS["next_week"], " ".join(validation.errors))

    def test_prepare_time_series_sorts_and_fills_missing_dates(self):
        sparse_df = self.df.iloc[[0, 2, 3, 5]].copy().sample(frac=1.0, random_state=7)

        prepared = prepare_time_series(
            sparse_df,
            date_column="order_date",
            target_column="sales",
            aggregation_frequency="D",
        )

        self.assertEqual(prepared["date"].iloc[0].date().isoformat(), "2025-01-01")
        self.assertEqual(prepared["date"].iloc[-1].date().isoformat(), "2025-01-06")
        self.assertEqual(len(prepared), 6)
        self.assertEqual(prepared.attrs["missing_timestamps"], 2)

    def test_validate_time_series_returns_structured_payload(self):
        validation = validate_time_series(self.df, date_column="order_date", target_column="sales")

        self.assertTrue(validation["valid"])
        self.assertEqual(validation["time_column"], "order_date")
        self.assertEqual(validation["target_column"], "sales")
        self.assertIn("cleaned_df", validation)

    def test_run_forecast_pipeline_returns_projection_and_recommendations(self):
        config = ForecastConfig(
            date_column="order_date",
            target_column="sales",
            driver_columns=["discount_rate", "region"],
            aggregation_frequency="D",
            horizon="next_month",
            model_strategy="hybrid",
            training_mode="local",
        )

        result = run_forecast_pipeline(self.df, config)

        self.assertEqual(result["status"], "PASSED")
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["forecast_table"]), 30)
        self.assertIn(result["trend_status"], {"growth", "stable", "volatility", "recovery_risk"})
        self.assertTrue(result["recommendations"])
        self.assertFalse(result["comparison_table"].empty)
        self.assertIn("artifact_payload", result)
        self.assertIn("data_score", result)
        self.assertIn("pipeline_trace", result)
        self.assertEqual(result["pipeline_trace"][3]["stage"], "forecast_ml")
        self.assertIn("model_metrics", result)
        self.assertIn("explanation", result)
        self.assertIn("data_quality", result)
        self.assertIn("model_quality", result)
        self.assertIn("risk", result)
        self.assertIn("reproducibility", result)
        self.assertIn("failure_patterns", result)
        self.assertIn("result_hash", result)
        self.assertIn("limitations", result)
        self.assertIn("decision_layer", result)
        self.assertIn("forecast", result)
        self.assertIn("time_series", result)
        self.assertIn("confidence", result)
        self.assertIn("forecast_metadata", result)
        self.assertIn("context", result)
        self.assertIn("suggestions", result)
        self.assertIn("recommended_next_step", result)
        self.assertIn("dashboard", result)
        self.assertIn("suggested_questions", result)
        self.assertIn("auto_config", result)
        self.assertEqual(result["auto_config"]["date_column"], "order_date")
        self.assertEqual(result["auto_config"]["target"], "sales")
        self.assertTrue(result["forecast"]["next_week"])
        self.assertIn("current_month", result["forecast"])
        self.assertIn("last_month", result["forecast"])
        self.assertTrue(result["time_series"])
        self.assertTrue(result["dashboard"]["charts"])
        self.assertEqual(result["visualization_type"], "line")
        self.assertTrue(result["decision_layer"]["decisions"])
        self.assertIn("learning_insights", result["decision_layer"])
        self.assertEqual(result["recommendations"][0]["recommended_action"], result["decision_layer"]["decisions"][0]["action"])
        self.assertIn("r2", result["model_metrics"])
        self.assertIn("top_features", result["explanation"])
        self.assertIn("decision_engine", [stage["stage"] for stage in result["pipeline_trace"]])

    def test_run_forecast_pipeline_returns_structured_error_when_date_column_missing(self):
        invalid_df = pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "region": ["North", "South", "East", "West"],
            }
        )

        result = run_forecast_pipeline(invalid_df, ForecastConfig(target_column="sales"))

        self.assertEqual(result["status"], "FAILED")
        self.assertIsInstance(result["error"], dict)
        self.assertIn("message", result["error"])
        self.assertIn("suggestion", result["error"])
        self.assertEqual(result["error"]["message"], "We couldn't detect a time column. Please select one.")
        self.assertIn("forecast_eligibility", result)
        self.assertFalse(result["forecast_eligibility"]["allowed"])
        self.assertEqual(result["forecast_eligibility"]["reason"], "No valid time column detected")
        self.assertIsNone(result["forecast_eligibility"]["detected_time_column"])
        self.assertTrue(result["forecast_eligibility"]["suggestions"])
        self.assertEqual(result["forecast"]["next_month"], [])
        self.assertFalse(result["dashboard"]["charts"])

    def test_run_forecast_pipeline_defaults_to_auto_configuration(self):
        result = run_forecast_pipeline(self.df, {})

        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["config"]["date_column"], "order_date")
        self.assertEqual(result["config"]["target_column"], "sales")
        self.assertEqual(result["auto_config"]["date_column"], "order_date")
        self.assertEqual(result["auto_config"]["target"], "sales")

    def test_run_forecast_pipeline_falls_back_to_trend_analysis_for_short_history(self):
        short_df = self.df.head(6).copy()
        config = ForecastConfig(
            date_column="order_date",
            target_column="sales",
            aggregation_frequency="D",
            horizon="next_month",
        )

        result = run_forecast_pipeline(short_df, config)

        self.assertEqual(result["status"], "FALLBACK")
        self.assertEqual(result["chosen_model"], "trend_analysis_fallback")
        self.assertTrue(result["forecast"]["next_month"])
        self.assertTrue(result["time_series"])
        self.assertTrue(result["dashboard"]["charts"])

    def test_persist_forecast_artifact_writes_object_store_payload(self):
        config = ForecastConfig(
            date_column="order_date",
            target_column="sales",
            driver_columns=["discount_rate"],
            aggregation_frequency="D",
            horizon="next_month",
        )
        result = run_forecast_pipeline(self.df, config)

        artifact_key, metadata = persist_forecast_artifact(
            result,
            workflow_id="wf-forecast",
            workflow_version=2,
            source_fingerprint="fingerprint-forecast",
            dataset_name="sales.csv",
        )

        stored_bytes = get_object_store().get_bytes(artifact_key)
        self.assertTrue(stored_bytes)
        self.assertEqual(metadata["workflow_id"], "wf-forecast")
        self.assertEqual(metadata["workflow_version"], 2)
        self.assertEqual(metadata["target_column"], "sales")
        self.assertEqual(metadata["horizon"], "next_month")
        self.assertEqual(metadata["result_hash"], result["result_hash"])
        self.assertIn("decision_layer", metadata)


if __name__ == "__main__":
    unittest.main()
