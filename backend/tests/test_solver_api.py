import importlib
import io
import os
import tempfile
import unittest
import zipfile
from unittest import mock

from fastapi.testclient import TestClient

from backend.aidssist_runtime.config import get_settings


class SolverApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_env = {
            "AIDSSIST_DATABASE_URL": os.getenv("AIDSSIST_DATABASE_URL"),
            "AIDSSIST_REDIS_URL": os.getenv("AIDSSIST_REDIS_URL"),
            "AIDSSIST_OBJECT_STORE_BACKEND": os.getenv("AIDSSIST_OBJECT_STORE_BACKEND"),
            "AIDSSIST_CORS_ORIGINS": os.getenv("AIDSSIST_CORS_ORIGINS"),
        }
        os.environ["AIDSSIST_DATABASE_URL"] = f"sqlite:///{self.temp_dir.name}/solver-api.sqlite3"
        os.environ["AIDSSIST_REDIS_URL"] = ""
        os.environ["AIDSSIST_OBJECT_STORE_BACKEND"] = "local"
        os.environ["AIDSSIST_CORS_ORIGINS"] = "http://localhost:5173"
        get_settings.cache_clear()

        import backend.aidssist_runtime.api as api_module
        import backend.aidssist_runtime.embedding as embedding_module
        import backend.aidssist_runtime.solver_orchestrator as solver_module

        self.api_module = importlib.reload(api_module)
        self.embedding_patcher = mock.patch.object(embedding_module, "genai", None)
        self.embedding_patcher.start()
        self.solver_llm_patcher = mock.patch.object(
            solver_module.prompt_pipeline,
            "_generate_groq_content",
            return_value=(
                '{"summary":"Hybrid redesign prepared.","solution_markdown":"### Proposed solution\\n'
                'Refactor the workspace into focused services and keep validator gates in front of each retry.",'
                '"redesign_recommendations":[{"title":"Split ingestion and reasoning","detail":"Keep file intake, retrieval, and reasoning isolated.","priority":"high"}],'
                '"validator_guidance":["Check deterministic logic first."],'
                '"implementation_outline":["ingestion","retrieval","validation"]}'
            ),
        )
        self.solver_llm_patcher.start()
        self.summary_patcher = mock.patch.object(
            solver_module.prompt_pipeline,
            "summarize_for_non_technical_user",
            return_value="Dataset analysis completed.",
        )
        self.summary_patcher.start()
        self.insights_patcher = mock.patch.object(
            solver_module.prompt_pipeline,
            "generate_insights",
            return_value="Review the highest performing region first.",
        )
        self.insights_patcher.start()
        self.client = TestClient(self.api_module.app)

    def tearDown(self):
        self.embedding_patcher.stop()
        self.solver_llm_patcher.stop()
        self.summary_patcher.stop()
        self.insights_patcher.stop()
        get_settings.cache_clear()
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()

    def _auth_headers(self):
        response = self.client.post(
            "/v1/auth/register",
            json={"email": "builder@aidssist.ai", "password": "supersecure", "display_name": "Builder"},
        )
        self.assertEqual(response.status_code, 200)
        return {"Authorization": f"Bearer {response.json()['token']}"}

    def test_workspace_asset_solve_transform_and_timeline_flow(self):
        headers = self._auth_headers()

        workspace_response = self.client.post(
            "/v1/workspaces",
            headers=headers,
            json={"name": "Ops Solver", "description": "Hybrid workspace"},
        )
        self.assertEqual(workspace_response.status_code, 200)
        workspace = workspace_response.json()

        csv_payload = (
            "date,sales,region\n"
            "2025-01-01,100,North\n"
            "2025-01-02,120,South\n"
            "2025-01-03,140,North\n"
        )
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("src/pipeline.py", "def transform(df):\n    return df\n")
            archive.writestr("README.md", "# Solver workspace\nThis project needs redesign guidance.\n")

        asset_response = self.client.post(
            "/v1/assets",
            headers=headers,
            data={"workspace_id": workspace["workspace_id"], "title": "Hybrid Intake"},
            files=[
                ("files", ("sales.csv", csv_payload.encode("utf-8"), "text/csv")),
                ("files", ("project.zip", archive_buffer.getvalue(), "application/zip")),
            ],
        )
        self.assertEqual(asset_response.status_code, 200, asset_response.text)
        asset = asset_response.json()
        self.assertEqual(asset["title"], "Hybrid Intake")
        self.assertGreaterEqual(asset["chunk_count"], 2)
        self.assertGreaterEqual(len(asset["datasets"]), 1)
        self.assertEqual(asset["datasets"][0]["columns"], ["date", "sales", "region"])
        self.assertEqual(asset["datasets"][0]["date_column"], "date")
        self.assertEqual(asset["datasets"][0]["target_column"], "sales")
        self.assertTrue(asset["datasets"][0]["forecast_eligibility"]["allowed"])
        self.assertEqual(asset["datasets"][0]["forecast_eligibility"]["detected_time_column"], "date")
        self.assertEqual(asset["datasets"][0]["stats"]["history_points"], 3)
        self.assertEqual(tuple(asset["datasets"][0]["stats"]["date_range"]), ("2025-01-01", "2025-01-03"))

        dataset_id = asset["datasets"][0]["dataset_id"]
        transform_response = self.client.post(
            f"/v1/datasets/{dataset_id}/transform",
            headers=headers,
            json={"instruction": "fill numeric missing with mean"},
        )
        self.assertEqual(transform_response.status_code, 200, transform_response.text)
        derived = transform_response.json()["derived_dataset"]
        self.assertEqual(derived["parent_dataset_id"], dataset_id)

        solve_response = self.client.post(
            "/v1/solve",
            headers=headers,
            json={
                "workspace_id": workspace["workspace_id"],
                "asset_id": asset["asset_id"],
                "dataset_id": dataset_id,
                "query": "Compare sales by region and explain the next best implementation steps.",
                "route_hint": "data",
            },
        )
        self.assertEqual(solve_response.status_code, 200, solve_response.text)
        run = solve_response.json()
        self.assertEqual(run["status"], "completed")
        self.assertTrue(run["validator_reports"])
        self.assertTrue(run["steps"])
        self.assertTrue(run["retrieval_trace"]["items"])
        pipeline_output = run["final_output"]["pipeline_output"]
        self.assertIn("analysis_contract", pipeline_output)
        self.assertIn("confidence", pipeline_output)
        self.assertTrue(pipeline_output["analysis_contract"]["recommendations"])
        self.assertIn("model_metrics", pipeline_output["analysis_contract"])
        self.assertIn("result_hash", pipeline_output["analysis_contract"])
        self.assertIn("limitations", pipeline_output["analysis_contract"])

        run_response = self.client.get(f"/v1/solve/{run['run_id']}", headers=headers)
        self.assertEqual(run_response.status_code, 200)
        self.assertEqual(run_response.json()["run_id"], run["run_id"])

        timeline_response = self.client.get(
            f"/v1/workspaces/{workspace['workspace_id']}/timeline",
            headers=headers,
        )
        self.assertEqual(timeline_response.status_code, 200)
        timeline = timeline_response.json()
        event_types = {item["event_type"] for item in timeline}
        self.assertIn("asset_uploaded", event_types)
        self.assertIn("derived_dataset", event_types)
        self.assertIn("solve_run", event_types)

    def test_demo_endpoint_returns_structured_payload(self):
        with mock.patch.object(
            self.api_module,
            "get_demo_payload",
            return_value={
                "dataset": {"metadata": {"row_count": 3}, "rows": [{"sales": 100}]},
                "datasets": [{"metadata": {"row_count": 3}, "rows": [{"sales": 100}]}],
                "queries": ["Predict next month sales", "Top 5 products", "Why revenue dropped"],
                "outputs": [
                    {"query": "Predict next month sales", "intent": "forecast", "output": {"summary": "ok"}},
                ],
                "dashboard": {"kpis": [{"metric": "total_sales", "value": 100}]},
                "stats": [{"label": "Sample rows", "value": 3, "detail": "Retail"}],
                "flow": [{"title": "Load sample dataset", "description": "Instant demo"}],
                "suggestions": [{"title": "Forecast next month sales", "prompt": "Forecast next month sales", "action_type": "forecast"}],
            },
        ):
            response = self.client.get("/demo")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["queries"]), 3)
        self.assertEqual(payload["outputs"][0]["intent"], "forecast")
        self.assertEqual(payload["stats"][0]["label"], "Sample rows")
        self.assertEqual(payload["flow"][0]["title"], "Load sample dataset")
        self.assertEqual(payload["datasets"][0]["metadata"]["row_count"], 3)

    def test_demo_data_endpoint_alias_returns_structured_payload(self):
        with mock.patch.object(
            self.api_module,
            "get_demo_payload",
            return_value={
                "dataset": {"metadata": {"row_count": 12}, "rows": [{"sales": 1200}]},
                "datasets": [{"metadata": {"row_count": 12}, "rows": [{"sales": 1200}]}],
                "queries": ["Predict next month sales"],
                "outputs": [{"query": "Predict next month sales", "intent": "forecast", "output": {"status": "PASSED"}}],
                "dashboard": {"kpis": [{"metric": "total_sales", "value": 16885}]},
                "stats": [{"label": "Revenue tracked", "value": 16885, "detail": "Precomputed"}],
                "flow": [{"title": "Open dashboard", "description": "Show KPI cards"}],
                "suggestions": [{"title": "Open forecast", "prompt": "Forecast next month sales", "action_type": "forecast"}],
            },
        ):
            response = self.client.get("/demo-data")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["dashboard"]["kpis"][0]["metric"], "total_sales")
        self.assertEqual(payload["outputs"][0]["output"]["status"], "PASSED")


if __name__ == "__main__":
    unittest.main()
