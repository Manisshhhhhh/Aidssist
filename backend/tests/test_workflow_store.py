import tempfile
import unittest
from pathlib import Path

from backend.workflow_store import WorkflowStore


class WorkflowStoreTests(unittest.TestCase):
    def test_save_workflow_creates_versions_and_lists_latest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = WorkflowStore(Path(temp_dir) / "workflows.sqlite3")

            workflow_v1 = store.save_workflow(
                name="Revenue Monitor",
                source_config={"kind": "csv", "file_name": "sales.csv", "snapshot_path": "/tmp/sales.csv"},
                cleaning_options={"trim_strings": True},
                analysis_query="show revenue by region",
                forecast_config={"date_column": "order_date", "target_column": "sales", "horizon": "next_month"},
                chart_preferences={"kind": "auto"},
                export_settings={"include_csv": True},
            )
            workflow_v2 = store.save_workflow(
                workflow_id=workflow_v1.workflow_id,
                name="Revenue Monitor",
                source_config={"kind": "csv", "file_name": "sales.csv", "snapshot_path": "/tmp/sales.csv"},
                cleaning_options={"trim_strings": False},
                analysis_query="show revenue by segment",
                forecast_config={"date_column": "order_date", "target_column": "sales", "horizon": "next_quarter"},
                chart_preferences={"kind": "bar"},
                export_settings={"include_csv": True},
            )

            latest_workflows = store.list_workflows()
            versions = store.list_workflow_versions(workflow_v1.workflow_id)
            store.close()

        self.assertEqual(workflow_v1.version, 1)
        self.assertEqual(workflow_v2.version, 2)
        self.assertEqual(len(latest_workflows), 1)
        self.assertEqual(latest_workflows[0].version, 2)
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0].analysis_query, "show revenue by segment")
        self.assertEqual(versions[0].forecast_config["horizon"], "next_quarter")

    def test_record_run_persists_audit_trail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = WorkflowStore(Path(temp_dir) / "workflows.sqlite3")
            workflow = store.save_workflow(
                name="Quality Check",
                source_config={"kind": "excel", "file_name": "sales.xlsx", "snapshot_path": "/tmp/sales.xlsx"},
                cleaning_options={"parse_dates": True},
                analysis_query="top 5 rows",
                forecast_config={"date_column": "order_date", "target_column": "sales", "horizon": "next_month"},
                chart_preferences={"kind": "auto"},
                export_settings={"include_csv": True, "include_json": True},
            )

            run_record = store.build_run_record(
                workflow_id=workflow.workflow_id,
                workflow_version=workflow.version,
                workflow_name=workflow.name,
                source_fingerprint="abc123",
                source_label="Excel file: sales.xlsx",
                validation_findings=[{"severity": "warning", "category": "high_missingness"}],
                cleaning_actions=["Trimmed whitespace in 'region'."],
                generated_code="result = df.head()",
                final_status="PASSED",
                error_message=None,
                export_artifacts=["csv", "json", "python"],
                analysis_query=workflow.analysis_query,
                result_summary="Looks healthy.",
                result_hash="hash-run-1",
            )
            store.record_run(run_record)
            runs = store.list_runs()
            store.close()

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].workflow_name, "Quality Check")
        self.assertEqual(runs[0].source_fingerprint, "abc123")
        self.assertEqual(runs[0].validation_findings[0]["category"], "high_missingness")
        self.assertEqual(runs[0].result_hash, "hash-run-1")

    def test_record_forecast_artifact_persists_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = WorkflowStore(Path(temp_dir) / "workflows.sqlite3")
            workflow = store.save_workflow(
                name="Forecast Monitor",
                source_config={"kind": "csv", "file_name": "sales.csv", "snapshot_path": "/tmp/sales.csv"},
                cleaning_options={"parse_dates": True},
                analysis_query="show revenue by region",
                forecast_config={"date_column": "order_date", "target_column": "sales", "horizon": "next_month"},
                chart_preferences={"kind": "line"},
                export_settings={"include_csv": True, "include_json": True},
            )

            artifact_record = store.build_forecast_artifact_record(
                workflow_id=workflow.workflow_id,
                workflow_version=workflow.version,
                workflow_name=workflow.name,
                source_fingerprint="fingerprint-1",
                source_label="CSV file: sales.csv",
                target_column="sales",
                horizon="next_month",
                model_name="ridge_regression",
                training_mode="local",
                status="PASSED",
                artifact_key="forecast_artifacts/wf-1/model.pkl",
                forecast_config={"date_column": "order_date", "target_column": "sales"},
                evaluation_metrics={"mape": 11.2},
                recommendation_payload=[{"title": "Lean into growth"}],
                summary="Sales are expected to grow.",
                result_hash="hash-forecast-1",
            )
            store.record_forecast_artifact(artifact_record)
            artifacts = store.list_forecast_artifacts(workflow_id=workflow.workflow_id)
            store.close()

        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].model_name, "ridge_regression")
        self.assertEqual(artifacts[0].target_column, "sales")
        self.assertEqual(artifacts[0].recommendation_payload[0]["title"], "Lean into growth")
        self.assertEqual(artifacts[0].result_hash, "hash-forecast-1")

    def test_dataset_and_job_lifecycle_round_trips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = WorkflowStore(Path(temp_dir) / "workflows.sqlite3")

            dataset = store.create_dataset(
                dataset_name="sales.csv",
                dataset_key="sales-key",
                source_fingerprint="fingerprint-123",
                source_kind="csv",
                source_label="CSV file: sales.csv",
                object_key="datasets/fingerprint-123/sales.csv",
                content_type="text/csv",
                size_bytes=1024,
            )
            fetched_dataset = store.get_dataset(dataset.dataset_id)

            job = store.create_job(
                dataset_id=dataset.dataset_id,
                query="top customers by revenue",
                intent="general",
                workflow_context={"workflow_id": "wf-1"},
                cache_key="analysis:cache-key",
            )
            running_job = store.mark_job_running(job.job_id)
            completed_job = store.complete_job(
                job.job_id,
                analysis_output={"result": {"kind": "text", "value": "ok"}},
                result_summary="Looks good.",
                cache_hit=True,
            )
            store.close()

        self.assertIsNotNone(fetched_dataset)
        self.assertEqual(fetched_dataset.dataset_name, "sales.csv")
        self.assertEqual(running_job.status, "running")
        self.assertEqual(completed_job.status, "completed")
        self.assertTrue(completed_job.cache_hit)
        self.assertEqual(completed_job.workflow_context["workflow_id"], "wf-1")
        self.assertEqual(completed_job.analysis_output["result"]["value"], "ok")

    def test_decision_history_round_trips_and_updates_outcome(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = WorkflowStore(Path(temp_dir) / "workflows.sqlite3")
            decision_record = store.build_decision_history_record(
                job_id="job-1",
                forecast_artifact_id="artifact-1",
                source_fingerprint="fingerprint-123",
                query="predict sales",
                decision={
                    "decision_id": "decision-1",
                    "action": "Increase focus on North",
                    "expected_impact": "Increase revenue by ~8%",
                    "confidence": "medium",
                    "risk_level": "low",
                    "priority": "HIGH",
                    "reasoning": "North leads current output.",
                },
                decision_confidence="medium",
                result_hash="hash-decision-1",
            )
            store.record_decision_history(decision_record)
            listed = store.list_decision_history(job_id="job-1")
            fetched = store.get_decision_history(decision_record.decision_history_id)
            updated = store.update_decision_outcome(
                decision_record.decision_history_id,
                "Outcome improved in the next cycle.",
            )
            store.close()

        self.assertEqual(len(listed), 1)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.decision_id, "decision-1")
        self.assertEqual(updated.outcome, "Outcome improved in the next cycle.")

    def test_failure_logs_and_matching_hash_queries_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = WorkflowStore(Path(temp_dir) / "workflows.sqlite3")
            dataset = store.create_dataset(
                dataset_name="sales.csv",
                dataset_key="sales-key",
                source_fingerprint="fingerprint-123",
                source_kind="csv",
                source_label="CSV file: sales.csv",
                object_key="datasets/fingerprint-123/sales.csv",
                content_type="text/csv",
                size_bytes=1024,
            )
            run_record = store.build_run_record(
                workflow_id="wf-1",
                workflow_version=1,
                workflow_name="Ops",
                source_fingerprint="fingerprint-123",
                source_label="CSV file: sales.csv",
                validation_findings=[],
                cleaning_actions=[],
                generated_code="result = 1",
                final_status="PASSED",
                error_message=None,
                export_artifacts=["json"],
                analysis_query="Predict sales",
                result_summary="ok",
                result_hash="hash-analysis",
            )
            store.record_run(run_record)
            solve_run = store.create_solve_run(
                workspace_id="ws-1",
                user_id=None,
                dataset_id=dataset.dataset_id,
                query="Predict sales",
                route="data",
            )
            store.complete_solve_run(
                solve_run.run_id,
                plan_text="plan",
                retrieval_trace={},
                retrieved_chunk_ids=[],
                final_output={"pipeline_output": {"result_hash": "hash-solve"}},
                final_summary="done",
                result_hash="hash-solve",
            )
            store.record_failure_log(query="Predict sales", error="boom", stage="execution", metadata={"job_id": "1"})
            run_hashes = store.list_matching_run_result_hashes(
                source_fingerprint="fingerprint-123",
                normalized_query="predict sales",
                analysis_intent="prediction",
            )
            solve_hashes = store.list_matching_solve_result_hashes(
                source_fingerprint="fingerprint-123",
                normalized_query="predict sales",
                route="data",
            )
            failures = store.list_failure_logs()
            store.close()

        self.assertEqual(run_hashes, ["hash-analysis"])
        self.assertEqual(solve_hashes, ["hash-solve"])
        self.assertEqual(len(failures), 1)


if __name__ == "__main__":
    unittest.main()
