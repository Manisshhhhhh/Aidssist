import importlib
import os
import tempfile
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from backend.aidssist_runtime.cache import get_cache_store
from backend.aidssist_runtime.config import get_settings
from backend.services.learning_engine import get_learning_patterns
from backend.workflow_store import WorkflowStore


class ApiSaaSTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "api.sqlite3")
        self.original_env = {
            "AIDSSIST_DATABASE_URL": os.getenv("AIDSSIST_DATABASE_URL"),
            "AIDSSIST_REDIS_URL": os.getenv("AIDSSIST_REDIS_URL"),
            "AIDSSIST_OBJECT_STORE_BACKEND": os.getenv("AIDSSIST_OBJECT_STORE_BACKEND"),
            "AIDSSIST_CORS_ORIGINS": os.getenv("AIDSSIST_CORS_ORIGINS"),
        }
        os.environ["AIDSSIST_DATABASE_URL"] = f"sqlite:///{self.db_path}"
        os.environ["AIDSSIST_REDIS_URL"] = ""
        os.environ["AIDSSIST_OBJECT_STORE_BACKEND"] = "local"
        os.environ["AIDSSIST_CORS_ORIGINS"] = "http://localhost:5173"
        get_settings.cache_clear()
        get_cache_store.cache_clear()

        import backend.aidssist_runtime.api as api_module

        self.api_module = importlib.reload(api_module)
        self.client = TestClient(self.api_module.app)

    def tearDown(self):
        get_cache_store.cache_clear()
        get_settings.cache_clear()
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()

    def test_register_me_and_history_endpoint(self):
        register_response = self.client.post(
            "/v1/auth/register",
            json={
                "email": "ceo@aidssist.ai",
                "password": "supersecure",
                "display_name": "CEO",
            },
        )
        self.assertEqual(register_response.status_code, 200)
        token = register_response.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        me_response = self.client.get("/v1/auth/me", headers=headers)
        self.assertEqual(me_response.status_code, 200)
        user_id = me_response.json()["user_id"]

        store = WorkflowStore()
        try:
            dataset = store.create_dataset(
                dataset_name="sales.csv",
                dataset_key="sales-key",
                source_fingerprint="source-1",
                source_kind="csv",
                source_label="CSV file: sales.csv",
                object_key="datasets/source-1/sales.csv",
                content_type="text/csv",
                size_bytes=512,
                user_id=user_id,
            )
            store.create_job(
                dataset_id=dataset.dataset_id,
                query="revenue by region",
                intent="general",
                workflow_context={},
                status="completed",
                analysis_output={"result": {"kind": "text", "value": "ok"}},
                result_summary="Revenue is concentrated in the East region.",
                user_id=user_id,
            )
        finally:
            store.close()

        history_response = self.client.get("/v1/history", headers=headers)
        self.assertEqual(history_response.status_code, 200)
        payload = history_response.json()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["dataset_name"], "sales.csv")
        self.assertEqual(payload[0]["query"], "revenue by region")

    def test_forecast_job_endpoint_returns_forecast_output(self):
        csv_payload = "order_date,sales,region,discount_rate\n"
        for index, date_value in enumerate(pd.date_range("2025-01-01", periods=40, freq="D")):
            csv_payload += (
                f"{date_value.date().isoformat()},{120 + index * 2},{'North' if index % 2 == 0 else 'South'},{0.05 if index % 3 == 0 else 0.08}\n"
            )

        upload_response = self.client.post(
            "/v1/uploads",
            files={"file": ("sales.csv", csv_payload.encode("utf-8"), "text/csv")},
        )
        self.assertEqual(upload_response.status_code, 200)
        dataset_id = upload_response.json()["dataset_id"]

        forecast_response = self.client.post(
            "/v1/jobs/forecast",
            json={
                "dataset_id": dataset_id,
                "forecast_config": {
                    "date_column": "order_date",
                    "target_column": "sales",
                    "aggregation_frequency": "D",
                    "horizon": "next_month",
                    "model_strategy": "hybrid",
                    "training_mode": "local",
                },
                "workflow_context": {"workflow_id": "wf-api-forecast"},
            },
        )
        self.assertEqual(forecast_response.status_code, 200)
        job_id = forecast_response.json()["job_id"]

        job_response = self.client.get(f"/v1/jobs/{job_id}")
        self.assertEqual(job_response.status_code, 200)
        payload = job_response.json()
        self.assertEqual(payload["intent"], "forecast")
        self.assertIsNone(payload["analysis_output"])
        self.assertIsNotNone(payload["forecast_output"])
        self.assertEqual(payload["forecast_output"]["status"], "PASSED")
        self.assertTrue(payload["forecast_output"]["forecast_eligibility"]["allowed"])
        self.assertEqual(payload["forecast_output"]["forecast_eligibility"]["detected_time_column"], "order_date")

    def test_upload_returns_auto_analysis_and_manual_endpoint_matches(self):
        csv_payload = (
            "report_date,region,confirmed_cases,deaths\n"
            "2025-02-01,North,12,1\n"
            "2025-02-02,North,14,1\n"
            "2025-02-01,South,20,2\n"
            "2025-02-02,South,18,2\n"
        )

        upload_response = self.client.post(
            "/v1/uploads",
            files={"file": ("medical.csv", csv_payload.encode("utf-8"), "text/csv")},
        )
        self.assertEqual(upload_response.status_code, 200)
        upload_payload = upload_response.json()
        dataset_id = upload_payload["dataset_id"]

        self.assertIn("auto_analysis", upload_payload)
        self.assertTrue(upload_payload["auto_analysis"]["tasks"])
        self.assertTrue(any("trend over time" in task.lower() for task in upload_payload["auto_analysis"]["tasks"]))

        manual_response = self.client.get(f"/v1/datasets/{dataset_id}/auto-analysis")
        self.assertEqual(manual_response.status_code, 200)
        manual_payload = manual_response.json()

        self.assertEqual(
            manual_payload["auto_analysis"]["tasks"],
            upload_payload["auto_analysis"]["tasks"],
        )
        self.assertTrue(manual_payload["auto_analysis"]["summary"])

    def test_decision_history_endpoints_are_user_scoped_and_support_outcomes(self):
        register_response = self.client.post(
            "/v1/auth/register",
            json={
                "email": "ops@aidssist.ai",
                "password": "supersecure",
                "display_name": "Ops",
            },
        )
        self.assertEqual(register_response.status_code, 200)
        token = register_response.json()["token"]
        user_id = register_response.json()["user"]["user_id"]
        headers = {"Authorization": f"Bearer {token}"}

        store = WorkflowStore()
        try:
            dataset = store.create_dataset(
                dataset_name="sales.csv",
                dataset_key="sales-key",
                source_fingerprint="source-decisions",
                source_kind="csv",
                source_label="CSV file: sales.csv",
                object_key="datasets/source-decisions/sales.csv",
                content_type="text/csv",
                size_bytes=512,
                user_id=user_id,
            )
            job = store.create_job(
                dataset_id=dataset.dataset_id,
                query="predict sales",
                intent="general",
                workflow_context={},
                status="completed",
                analysis_output={"result": {"kind": "text", "value": "ok"}},
                result_summary="Prediction completed.",
                user_id=user_id,
            )
            decision = store.build_decision_history_record(
                job_id=job.job_id,
                forecast_artifact_id=None,
                source_fingerprint=dataset.source_fingerprint,
                query=job.query,
                decision={
                    "decision_id": "decision-api-1",
                    "action": "Expand the North region",
                    "expected_impact": "Increase revenue by ~10%",
                    "confidence": "high",
                    "risk_level": "low",
                    "priority": "HIGH",
                    "reasoning": "North is leading current output.",
                },
                decision_confidence="high",
                result_hash="hash-api-1",
            )
            store.record_decision_history(decision)
            stale_patterns = get_learning_patterns(store, dataset.source_fingerprint, refresh=True)
            self.assertEqual(stale_patterns, {})
        finally:
            store.close()

        list_response = self.client.get("/v1/decisions", headers=headers)
        self.assertEqual(list_response.status_code, 200)
        decisions = list_response.json()
        self.assertEqual(len(decisions), 1)
        decision_history_id = decisions[0]["decision_history_id"]
        self.assertEqual(decisions[0]["decision_json"]["action"], "Expand the North region")

        detail_response = self.client.get(f"/v1/decisions/{decision_history_id}", headers=headers)
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(detail_response.json()["decision_id"], "decision-api-1")

        update_response = self.client.patch(
            f"/v1/decisions/{decision_history_id}/outcome",
            json={"outcome": "North outperformed in the next month."},
            headers=headers,
        )
        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(update_response.json()["outcome"], "North outperformed in the next month.")

        store = WorkflowStore()
        try:
            refreshed_patterns = get_learning_patterns(store, "source-decisions")
        finally:
            store.close()
        self.assertIn("expand_region", refreshed_patterns)
        self.assertEqual(refreshed_patterns["expand_region"]["sample_size"], 1)


if __name__ == "__main__":
    unittest.main()
