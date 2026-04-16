import unittest
from contextlib import ExitStack
from unittest import mock

import pandas as pd

from backend import prompt_pipeline


test_cases = [
    "Predict next month sales",
    "Why revenue dropped",
    "Top 5 products by rating",
    "Compare region performance",
    "Forecast without date column",
    "Prediction with missing target",
]


class RealWorldHarnessTests(unittest.TestCase):
    def setUp(self):
        self.sales_df = pd.DataFrame(
            {
                "order_date": pd.date_range("2025-01-01", periods=16, freq="D"),
                "sales": [100, 105, 102, 110, 115, 118, 117, 121, 125, 129, 132, 135, 138, 142, 145, 149],
                "revenue": [1000, 1010, 990, 980, 970, 960, 955, 950, 940, 935, 930, 925, 920, 918, 915, 910],
                "region": ["North", "South", "East", "West"] * 4,
                "product": ["A", "B", "C", "D"] * 4,
            }
        )
        self.ratings_df = pd.DataFrame(
            {
                "product": ["A", "B", "C", "D", "E", "F"],
                "rating": [4.9, 4.2, 4.7, 3.8, 4.5, 4.1],
                "region": ["North", "South", "East", "West", "North", "South"],
            }
        )
        self.no_date_df = self.sales_df.drop(columns=["order_date"]).copy()
        self.missing_target_df = pd.DataFrame(
            {
                "region": ["North", "South", "East"],
                "segment": ["SMB", "Enterprise", "Mid-Market"],
            }
        )

    @staticmethod
    def _patched_summary(*args, **kwargs):
        del args, kwargs
        return "summary"

    @staticmethod
    def _patched_insights(*args, **kwargs):
        del args, kwargs
        return "insights"

    def _evaluate_output(
        self,
        query: str,
        output: dict,
        *,
        expected_status: str,
        expected_error_tokens: tuple[str, ...] = (),
    ) -> dict:
        warnings = list(output.get("warnings") or [])
        failures = list(output.get("failure_events") or [])
        incorrect_outputs: list[str] = []

        if output.get("test_status") != expected_status:
            incorrect_outputs.append(
                f"Expected status {expected_status}, received {output.get('test_status')}"
            )
        if not output.get("recommendations"):
            incorrect_outputs.append("Recommendations were not generated.")
        contract = dict(output.get("analysis_contract") or {})
        for required_key in ("data_quality", "model_quality", "risk", "reproducibility", "decision_layer"):
            if required_key not in contract:
                incorrect_outputs.append(f"Missing contract key: {required_key}")
        if expected_status == "PASSED" and not ((contract.get("decision_layer") or {}).get("decisions") or []):
            incorrect_outputs.append("Structured decisions were not generated.")

        haystacks = [str(output.get("error") or "").lower()]
        haystacks.extend(str(event).lower() for event in failures)
        for token in expected_error_tokens:
            if not any(token in haystack for haystack in haystacks):
                incorrect_outputs.append(f"Expected error token not found: {token}")

        return {
            "query": query,
            "status": output.get("test_status"),
            "warnings": warnings,
            "failures": failures,
            "incorrect_outputs": incorrect_outputs,
        }

    def _run_general_case(
        self,
        query: str,
        df: pd.DataFrame,
        *,
        expected_status: str,
        report_query: str | None = None,
    ) -> dict:
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"))
            stack.enter_context(mock.patch.object(prompt_pipeline, "log_step"))
            stack.enter_context(
                mock.patch.object(
                    prompt_pipeline,
                    "summarize_for_non_technical_user",
                    side_effect=self._patched_summary,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    prompt_pipeline,
                    "generate_insights",
                    side_effect=self._patched_insights,
                )
            )
            output = prompt_pipeline.run_builder_pipeline(query, df, workflow_context={"source_fingerprint": "fp-general"})
        return self._evaluate_output(report_query or query, output, expected_status=expected_status)

    def test_real_world_harness_runs_pipeline_cases_and_captures_issues(self):
        reports: list[dict] = []

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"))
            stack.enter_context(mock.patch.object(prompt_pipeline, "log_step"))
            stack.enter_context(mock.patch.object(prompt_pipeline, "validate_forecast_dataset", return_value="VALID\nForecast data looks usable."))
            stack.enter_context(mock.patch.object(prompt_pipeline, "generate_forecast_prep_code", return_value="result = prepared_df"))
            stack.enter_context(mock.patch.object(prompt_pipeline, "generate_sales_prediction_code", return_value="result = forecast_df"))
            stack.enter_context(
                mock.patch.object(
                    prompt_pipeline,
                    "execute_code",
                    side_effect=[
                        (self.sales_df[["order_date", "sales", "region"]].copy(), None),
                        (
                            pd.DataFrame(
                                {
                                    "prediction_step": [1, 2, 3],
                                    "prediction_target": ["sales", "sales", "sales"],
                                    "predicted_value": [152.0, 156.0, 160.0],
                                }
                            ),
                            None,
                        ),
                    ],
                )
            )
            stack.enter_context(mock.patch.object(prompt_pipeline, "summarize_for_non_technical_user", side_effect=self._patched_summary))
            stack.enter_context(mock.patch.object(prompt_pipeline, "generate_insights", side_effect=self._patched_insights))
            output = prompt_pipeline.run_builder_pipeline(
                "Predict next month sales",
                self.sales_df,
                workflow_context={"source_fingerprint": "fp-sales"},
            )
            reports.append(self._evaluate_output("Predict next month sales", output, expected_status="PASSED"))

        reports.append(self._run_general_case("Why revenue dropped", self.sales_df, expected_status="PASSED"))

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"))
            stack.enter_context(mock.patch.object(prompt_pipeline, "log_step"))
            stack.enter_context(mock.patch.object(prompt_pipeline, "validate_ratings_dataset", return_value="VALID\nRatings data looks usable."))
            stack.enter_context(mock.patch.object(prompt_pipeline, "generate_ratings_analysis_code", return_value="result = ratings"))
            stack.enter_context(
                mock.patch.object(
                    prompt_pipeline,
                    "execute_code",
                    return_value=(
                        self.ratings_df.sort_values("rating", ascending=False).head(5).reset_index(drop=True),
                        None,
                    ),
                )
            )
            stack.enter_context(mock.patch.object(prompt_pipeline, "summarize_for_non_technical_user", side_effect=self._patched_summary))
            stack.enter_context(mock.patch.object(prompt_pipeline, "generate_insights", side_effect=self._patched_insights))
            output = prompt_pipeline.run_builder_pipeline(
                "Top 5 products by rating",
                self.ratings_df,
                workflow_context={"source_fingerprint": "fp-ratings"},
            )
            reports.append(self._evaluate_output("Top 5 products by rating", output, expected_status="PASSED"))

        reports.append(
            self._run_general_case(
                "Compare sales by region",
                self.sales_df,
                expected_status="PASSED",
                report_query="Compare region performance",
            )
        )

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"))
            stack.enter_context(mock.patch.object(prompt_pipeline, "log_step"))
            stack.enter_context(mock.patch.object(prompt_pipeline, "validate_forecast_dataset", return_value="VALID\nForecast data looks usable."))
            stack.enter_context(mock.patch.object(prompt_pipeline, "generate_forecast_prep_code", return_value="result = df.copy()"))
            stack.enter_context(mock.patch.object(prompt_pipeline, "fix_code", return_value="result = df.copy()"))
            stack.enter_context(
                mock.patch.object(
                    prompt_pipeline,
                    "execute_code",
                    side_effect=[
                        (None, "ValueError: missing date column"),
                        (None, "ValueError: missing date column"),
                        (None, "ValueError: missing date column"),
                    ],
                )
            )
            output = prompt_pipeline.run_builder_pipeline(
                "Forecast without date column",
                self.no_date_df,
                workflow_context={"source_fingerprint": "fp-no-date"},
            )
            reports.append(
                self._evaluate_output(
                    "Forecast without date column",
                    output,
                    expected_status="FAILED",
                    expected_error_tokens=("missing date column",),
                )
            )

        reports.append(
            self._run_general_case(
                "Estimate next month demand",
                self.missing_target_df,
                expected_status="PASSED",
            )
        )
        reports[-1]["query"] = "Prediction with missing target"

        self.assertEqual([report["query"] for report in reports], test_cases)
        incorrect = [report for report in reports if report["incorrect_outputs"]]
        self.assertFalse(incorrect, incorrect)

        by_query = {report["query"]: report for report in reports}
        self.assertEqual(by_query["Forecast without date column"]["status"], "FAILED")
        self.assertTrue(by_query["Forecast without date column"]["failures"])
        self.assertEqual(by_query["Prediction with missing target"]["status"], "PASSED")


if __name__ == "__main__":
    unittest.main()
