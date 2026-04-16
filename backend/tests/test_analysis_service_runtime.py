import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from backend.aidssist_runtime.analysis_service import (
    build_result_artifact,
    create_dataset_from_upload,
    detect_date_column,
    detect_target_column,
    extract_schema,
    get_dataset_summary,
    submit_analysis_job,
    submit_forecast_job,
)
from backend.aidssist_runtime.cache import get_cache_store
from backend.aidssist_runtime.config import get_settings
from backend.aidssist_runtime.storage import get_object_store
from backend.workflow_store import WorkflowStore


class AnalysisServiceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "runtime.sqlite3")
        self.original_env = {
            "AIDSSIST_DATABASE_URL": os.getenv("AIDSSIST_DATABASE_URL"),
            "AIDSSIST_REDIS_URL": os.getenv("AIDSSIST_REDIS_URL"),
            "AIDSSIST_OBJECT_STORE_BACKEND": os.getenv("AIDSSIST_OBJECT_STORE_BACKEND"),
        }
        os.environ["AIDSSIST_DATABASE_URL"] = f"sqlite:///{self.db_path}"
        os.environ["AIDSSIST_REDIS_URL"] = ""
        os.environ["AIDSSIST_OBJECT_STORE_BACKEND"] = "local"
        get_settings.cache_clear()
        get_cache_store.cache_clear()
        get_object_store.cache_clear()

        self.csv_bytes = (
            b"order_date,sales,region\n"
            b"2025-01-01,100,North\n"
            b"2025-01-02,125,South\n"
            b"2025-01-03,140,West\n"
        )

    def tearDown(self):
        get_cache_store.cache_clear()
        get_object_store.cache_clear()
        get_settings.cache_clear()
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()

    @staticmethod
    def _analysis_output(query: str):
        result_df = pd.DataFrame(
            {
                "period": ["next_week", "next_month"],
                "prediction": [410.0, 1625.0],
            }
        )
        return {
            "query": query,
            "intent": "forecast",
            "analysis_contract": {
                "intent": "prediction",
                "tool_used": "PYTHON",
                "analysis_mode": "prediction",
                "execution_plan": {
                    "sql_plan": None,
                    "python_steps": ["validate forecast inputs", "run python forecast"],
                    "excel_logic": {},
                    "fallback_reason": None,
                },
                "code": "result = df.head()",
                "result_summary": "Forecast generated successfully.",
                "insights": ["Forecast looks stable."],
                "recommendations": ["Validate the result before rollout."],
                "confidence": "7/10",
                "warnings": [],
            },
            "build_query": query,
            "build_plan": "Validate, prepare, predict.",
            "module_validation": "VALID\nForecast dataset is suitable.",
            "generated_code": "result = df.head()",
            "result": result_df,
            "test_status": "PASSED",
            "test_error": None,
            "error": None,
            "summary": "Forecast generated successfully.",
            "insights": "Forecast looks stable.",
            "tool_used": "PYTHON",
            "analysis_mode": "prediction",
            "execution_plan": {
                "sql_plan": None,
                "python_steps": ["validate forecast inputs", "run python forecast"],
                "excel_logic": {},
                "fallback_reason": None,
            },
            "recommendations": ["Validate the result before rollout."],
            "warnings": [],
            "confidence": "7/10",
            "business_decisions": "Validate the result before rollout.",
            "workflow_context": {"workflow_id": "wf-runtime"},
        }

    def test_create_dataset_and_sync_job_processing(self):
        dataset = create_dataset_from_upload("sales.csv", self.csv_bytes, "text/csv")

        with mock.patch(
            "backend.aidssist_runtime.analysis_service.run_builder_pipeline",
            return_value=self._analysis_output("predict sales for next month"),
        ) as pipeline_mock:
            job = submit_analysis_job(
                dataset.dataset_id,
                "predict sales for next month",
                workflow_context={"workflow_id": "wf-runtime", "export_settings": {"include_csv": True}},
            )

        self.assertEqual(job.status, "completed")
        self.assertFalse(job.cache_hit)
        self.assertIsNotNone(job.analysis_output)
        self.assertEqual(job.analysis_output["summary"], "Forecast generated successfully.")
        self.assertEqual(job.analysis_output["analysis_contract"]["intent"], "prediction")
        self.assertEqual(job.analysis_output["analysis_contract"]["tool_used"], "PYTHON")
        self.assertEqual(job.analysis_output["analysis_contract"]["analysis_mode"], "prediction")
        self.assertIn("system_decision", job.analysis_output["analysis_contract"])
        self.assertIn("selected_mode", job.analysis_output["analysis_contract"]["system_decision"])
        self.assertTrue(job.analysis_output["analysis_contract"]["execution_plan"])
        self.assertTrue(any(step["tool"] == "PYTHON" for step in job.analysis_output["analysis_contract"]["execution_plan"]))
        self.assertTrue(job.analysis_output["analysis_contract"]["execution_trace"])
        self.assertEqual(job.analysis_output["tool_used"], "PYTHON")
        self.assertEqual(job.analysis_output["analysis_mode"], "prediction")
        self.assertTrue(job.analysis_output["execution_trace"])
        self.assertIn("optimization", job.analysis_output["analysis_contract"])
        self.assertIn("optimization", job.analysis_output)
        self.assertEqual(job.analysis_output["confidence"], "7/10")
        self.assertIn("data_quality", job.analysis_output["analysis_contract"])
        self.assertIn("model_metrics", job.analysis_output["analysis_contract"])
        self.assertIn("explanation", job.analysis_output["analysis_contract"])
        self.assertIn("model_quality", job.analysis_output["analysis_contract"])
        self.assertIn("risk", job.analysis_output["analysis_contract"])
        self.assertIn("reproducibility", job.analysis_output["analysis_contract"])
        self.assertIn("result_hash", job.analysis_output["analysis_contract"])
        self.assertIn("limitations", job.analysis_output["analysis_contract"])
        self.assertIn("decision_layer", job.analysis_output["analysis_contract"])
        self.assertIn("context", job.analysis_output["analysis_contract"])
        self.assertIn("suggestions", job.analysis_output["analysis_contract"])
        self.assertIn("recommended_next_step", job.analysis_output["analysis_contract"])
        self.assertTrue(job.analysis_output["decision_layer"]["decisions"])
        self.assertIn("learning_insights", job.analysis_output["decision_layer"])
        self.assertEqual(job.analysis_output["cache_status"]["status"], "stored")
        self.assertEqual(job.analysis_output["memory_update"]["status"], "recorded")
        self.assertTrue(job.analysis_output["pipeline_trace"])
        pipeline_mock.assert_called_once()

        payload, media_type, file_name = build_result_artifact(job)
        self.assertEqual(media_type, "text/csv")
        self.assertEqual(file_name, "analysis_result.csv")
        self.assertIn(b"period,prediction", payload)

        store = WorkflowStore(self.db_path)
        try:
            runs = store.list_runs(limit=5)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].workflow_id, "wf-runtime")
            self.assertEqual(runs[0].final_status, "PASSED")
            self.assertEqual(runs[0].result_hash, job.analysis_output["result_hash"])
        finally:
            store.close()

    def test_get_dataset_summary_includes_schema_and_auto_detected_sales_mapping(self):
        dataset = create_dataset_from_upload("sales.csv", self.csv_bytes, "text/csv")

        summary = get_dataset_summary(dataset.dataset_id)

        self.assertEqual(summary["columns"], ["order_date", "sales", "region"])
        self.assertEqual(summary["date_column"], "order_date")
        self.assertEqual(summary["target_column"], "sales")
        self.assertEqual(summary["auto_config"]["date_column"], "order_date")
        self.assertEqual(summary["auto_config"]["target"], "sales")
        self.assertIn(summary["auto_config"]["confidence"], {"high", "medium"})
        self.assertTrue(summary["forecast_eligibility"]["allowed"])
        self.assertEqual(summary["forecast_eligibility"]["detected_time_column"], "order_date")
        self.assertEqual(summary["stats"]["history_points"], 3)
        self.assertEqual(tuple(summary["stats"]["date_range"]), ("2025-01-01", "2025-01-03"))
        self.assertEqual(summary["dtypes"]["sales"], "int64")
        self.assertIn("What affects sales?", summary["suggested_questions"])

    def test_get_dataset_summary_detects_covid_style_dataset(self):
        covid_bytes = (
            b"report_date,cases,deaths,region\n"
            b"2020-03-01,12,1,North\n"
            b"2020-03-02,18,1,North\n"
            b"2020-03-03,25,2,South\n"
        )
        dataset = create_dataset_from_upload("covid.csv", covid_bytes, "text/csv")

        summary = get_dataset_summary(dataset.dataset_id)

        self.assertEqual(summary["date_column"], "report_date")
        self.assertEqual(summary["target_column"], "cases")
        self.assertTrue(summary["forecast_eligibility"]["allowed"])
        self.assertEqual(tuple(summary["stats"]["date_range"]), ("2020-03-01", "2020-03-03"))

    def test_get_dataset_summary_blocks_non_time_dataset_for_forecasting(self):
        non_time_bytes = (
            b"sales,profit,region\n"
            b"100,22,North\n"
            b"120,28,South\n"
            b"140,31,West\n"
        )
        dataset = create_dataset_from_upload("not-time.csv", non_time_bytes, "text/csv")

        summary = get_dataset_summary(dataset.dataset_id)

        self.assertIsNone(summary["date_column"])
        self.assertEqual(summary["target_column"], "sales")
        self.assertFalse(summary["forecast_eligibility"]["allowed"])
        self.assertEqual(summary["forecast_eligibility"]["reason"], "No valid time column detected")
        self.assertIsNone(summary["forecast_eligibility"]["detected_time_column"])
        self.assertIn("Use analysis mode instead", summary["forecast_eligibility"]["suggestions"])

    def test_detect_date_column_returns_none_when_dataset_has_no_date_field(self):
        df = pd.DataFrame(
            {
                "sales": [100, 120, 140],
                "region": ["North", "South", "West"],
            }
        )

        self.assertIsNone(detect_date_column(df))
        self.assertEqual(detect_target_column(df), "sales")

    def test_extract_schema_and_detection_handle_mixed_types(self):
        df = pd.DataFrame(
            {
                "report_date": ["2025-01-01", "2025-01-02", None, "2025-01-04"],
                "sales": [100.0, 120.5, 141.0, 150.25],
                "notes": ["launch", 7, "backfill", None],
            }
        )

        schema = extract_schema(df)

        self.assertEqual([str(column) for column in schema["columns"]], ["report_date", "sales", "notes"])
        self.assertEqual(schema["dtypes"]["sales"], "float64")
        self.assertEqual(detect_date_column(df), "report_date")
        self.assertEqual(detect_target_column(df), "sales")

    def test_cached_replay_marks_cache_hit_and_skips_pipeline(self):
        dataset = create_dataset_from_upload("sales.csv", self.csv_bytes, "text/csv")

        with mock.patch(
            "backend.aidssist_runtime.analysis_service.run_builder_pipeline",
            return_value=self._analysis_output("predict sales for next month"),
        ) as pipeline_mock:
            first_job = submit_analysis_job(dataset.dataset_id, "predict sales for next month")
            second_job = submit_analysis_job(dataset.dataset_id, "predict sales for next month")

        self.assertEqual(first_job.status, "completed")
        self.assertEqual(second_job.status, "completed")
        self.assertFalse(first_job.cache_hit)
        self.assertTrue(second_job.cache_hit)
        self.assertEqual(second_job.analysis_output["cache_status"]["status"], "hit")
        self.assertEqual(second_job.analysis_output["memory_update"]["status"], "replayed")
        self.assertEqual(second_job.analysis_output["analysis_contract"]["tool_used"], "PYTHON")
        self.assertEqual(second_job.analysis_output["analysis_contract"]["analysis_mode"], "prediction")
        self.assertEqual(second_job.analysis_output["tool_used"], "PYTHON")
        self.assertEqual(second_job.analysis_output["analysis_mode"], "prediction")
        self.assertTrue(second_job.analysis_output["analysis_contract"]["execution_plan"])
        self.assertTrue(any(step["tool"] == "PYTHON" for step in second_job.analysis_output["analysis_contract"]["execution_plan"]))
        self.assertTrue(second_job.analysis_output["analysis_contract"]["execution_trace"])
        self.assertIn("optimization", second_job.analysis_output["analysis_contract"])
        self.assertIn("data_quality", second_job.analysis_output["analysis_contract"])
        self.assertIn("model_metrics", second_job.analysis_output["analysis_contract"])
        self.assertIn("limitations", second_job.analysis_output["analysis_contract"])
        self.assertIn("decision_layer", second_job.analysis_output["analysis_contract"])
        self.assertTrue(second_job.analysis_output["decision_layer"]["decisions"])
        self.assertIn("learning_insights", second_job.analysis_output["decision_layer"])
        pipeline_mock.assert_called_once()

    def test_submit_forecast_job_processes_and_records_artifact(self):
        forecast_bytes = "order_date,sales,region,discount_rate\n".encode("utf-8")
        for day_offset, day_value in enumerate(pd.date_range("2025-01-01", periods=48, freq="D")):
            forecast_bytes += (
                f"{day_value.date().isoformat()},{100 + day_offset * 3},{'North' if day_offset % 2 == 0 else 'South'},{0.05 if day_offset % 3 == 0 else 0.08}\n"
            ).encode("utf-8")

        dataset = create_dataset_from_upload("sales.csv", forecast_bytes, "text/csv")

        job = submit_forecast_job(
            dataset.dataset_id,
            {
                "date_column": "order_date",
                "target_column": "sales",
                "aggregation_frequency": "D",
                "horizon": "next_month",
                "model_strategy": "hybrid",
                "training_mode": "local",
            },
            workflow_context={"workflow_id": "wf-forecast-runtime", "workflow_name": "Forecast Runtime"},
        )

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.intent, "forecast")
        self.assertIsNotNone(job.analysis_output)
        self.assertIn(job.analysis_output["chosen_model"], {"linear_regression", "ridge_regression", "moving_average", "naive_last_value", "random_forest"})
        self.assertIn("artifact_metadata", job.analysis_output)
        self.assertIn("data_quality", job.analysis_output)
        self.assertIn("model_metrics", job.analysis_output)
        self.assertIn("explanation", job.analysis_output)
        self.assertIn("model_quality", job.analysis_output)
        self.assertIn("risk", job.analysis_output)
        self.assertIn("reproducibility", job.analysis_output)
        self.assertIn("result_hash", job.analysis_output)
        self.assertIn("limitations", job.analysis_output)
        self.assertIn("decision_layer", job.analysis_output)
        self.assertIn("forecast_metadata", job.analysis_output)
        self.assertIn("context", job.analysis_output)
        self.assertIn("suggestions", job.analysis_output)
        self.assertIn("recommended_next_step", job.analysis_output)
        self.assertIn("dashboard", job.analysis_output)
        self.assertIn("suggested_questions", job.analysis_output)
        self.assertIn("active_filter", job.analysis_output)
        self.assertIn("visualization_type", job.analysis_output)
        self.assertTrue(job.analysis_output["decision_layer"]["decisions"])
        self.assertIn("learning_insights", job.analysis_output["decision_layer"])

        store = WorkflowStore(self.db_path)
        try:
            artifacts = store.list_forecast_artifacts(workflow_id="wf-forecast-runtime")
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].target_column, "sales")
            self.assertEqual(artifacts[0].horizon, "next_month")
            self.assertEqual(artifacts[0].result_hash, job.analysis_output["result_hash"])
            decisions = store.list_decision_history(job_id=job.job_id)
            self.assertTrue(decisions)
            self.assertEqual(decisions[0].forecast_artifact_id, job.analysis_output["artifact_metadata"]["artifact_id"])
        finally:
            store.close()

    def test_submit_forecast_job_uses_auto_configuration_when_config_is_empty(self):
        forecast_bytes = "order_date,sales,region,discount_rate\n".encode("utf-8")
        for day_offset, day_value in enumerate(pd.date_range("2025-01-01", periods=48, freq="D")):
            forecast_bytes += (
                f"{day_value.date().isoformat()},{100 + day_offset * 2},{'North' if day_offset % 2 == 0 else 'South'},{0.05 if day_offset % 3 == 0 else 0.08}\n"
            ).encode("utf-8")

        dataset = create_dataset_from_upload("sales.csv", forecast_bytes, "text/csv")

        job = submit_forecast_job(
            dataset.dataset_id,
            {},
            workflow_context={"workflow_id": "wf-forecast-auto-runtime"},
        )

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.intent, "forecast")
        self.assertEqual(job.analysis_output["config"]["date_column"], "order_date")
        self.assertEqual(job.analysis_output["config"]["target_column"], "sales")
        self.assertEqual(job.analysis_output["auto_config"]["date_column"], "order_date")
        self.assertEqual(job.analysis_output["auto_config"]["target"], "sales")

    def test_submit_forecast_job_returns_structured_block_for_non_time_dataset(self):
        dataset = create_dataset_from_upload(
            "customers.csv",
            b"sales,profit,segment\n100,20,A\n120,25,B\n140,28,A\n",
            "text/csv",
        )

        job = submit_forecast_job(
            dataset.dataset_id,
            {},
            workflow_context={"workflow_id": "wf-forecast-blocked-runtime"},
        )

        self.assertEqual(job.status, "completed")
        self.assertEqual(job.intent, "forecast")
        self.assertEqual(job.analysis_output["status"], "FAILED")
        self.assertFalse(job.analysis_output["forecast_eligibility"]["allowed"])
        self.assertEqual(job.analysis_output["forecast_eligibility"]["reason"], "No valid time column detected")
        self.assertEqual(job.analysis_output["error"]["message"], "We couldn't detect a time column. Please select one.")

    def test_analysis_failures_are_logged_without_interrupting_job_failure_flow(self):
        dataset = create_dataset_from_upload("sales.csv", self.csv_bytes, "text/csv")

        with mock.patch(
            "backend.aidssist_runtime.analysis_service.run_builder_pipeline",
            side_effect=RuntimeError("pipeline exploded"),
        ):
            with self.assertRaises(RuntimeError):
                submit_analysis_job(dataset.dataset_id, "predict sales for next month")

        store = WorkflowStore(self.db_path)
        try:
            failures = store.list_failure_logs(stage="analysis_service")
            self.assertEqual(len(failures), 1)
            self.assertIn("pipeline exploded", failures[0].error_message)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
