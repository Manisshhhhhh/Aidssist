import unittest
from unittest import mock

import pandas as pd

from backend import prompt_pipeline


class ProviderRoutingTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "region": ["North", "South"],
                "sales": [100, 200],
            }
        )
        self.forecast_df = pd.DataFrame(
            {
                "order_date": pd.date_range("2024-01-01", periods=14, freq="D"),
                "sales": [120, 126, 129, 134, 138, 142, 145, 149, 152, 156, 160, 164, 169, 172],
            }
        )

    @staticmethod
    def _find_plan_step(execution_plan, tool):
        for step in execution_plan or []:
            if step.get("tool") == tool:
                return step
        return None

    def test_generate_code_uses_gemini_provider(self):
        with mock.patch.object(prompt_pipeline, "_generate_groq_content", return_value="result = 1") as compatibility_mock:
            with mock.patch.object(prompt_pipeline, "_generate_gemini_content", return_value="unused") as gemini_mock:
                result = prompt_pipeline.generate_code("top rows", self.df)

        self.assertEqual(result, "result = 1")
        compatibility_mock.assert_called_once()
        gemini_mock.assert_not_called()

    def test_rewrite_query_uses_plain_gemini(self):
        with mock.patch.object(prompt_pipeline, "_generate_gemini_content", return_value="better query") as gemini_mock:
            result = prompt_pipeline.rewrite_query("sales?", self.df)

        self.assertEqual(result, "better query")
        self.assertFalse(gemini_mock.call_args.kwargs["use_search"])

    def test_generate_insights_uses_dataset_grounded_gemini_prompt(self):
        with mock.patch.object(prompt_pipeline, "_generate_gemini_content", return_value="insight") as gemini_mock:
            result = prompt_pipeline.generate_insights("top customers by revenue", {"sales": 100}, self.df)

        self.assertEqual(result, "insight")
        self.assertFalse(gemini_mock.call_args.kwargs["use_search"])
        prompt_text = gemini_mock.call_args.kwargs["prompt"]
        self.assertIn("top customers by revenue", prompt_text)
        self.assertIn("region", prompt_text)
        self.assertIn("sales", prompt_text)
        self.assertIn("Rank insights by business impact", prompt_text)
        self.assertIn("🔑 Key Insight:", prompt_text)
        self.assertIn("⚠️ Risk / Opportunity:", prompt_text)
        self.assertIn("🎯 Recommended Action:", prompt_text)

    def test_fix_code_uses_gemini_compatibility_path(self):
        with mock.patch.object(prompt_pipeline, "_generate_groq_content", return_value="result = 2") as compatibility_mock:
            result = prompt_pipeline.fix_code("bad", "NameError: x", self.df, "fix it")

        self.assertEqual(result, "result = 2")
        compatibility_mock.assert_called_once()

    def test_safe_execute_allows_validation_exceptions(self):
        result, error = prompt_pipeline.safe_execute("raise ValueError('missing sales column')", self.df)

        self.assertIsNone(result)
        self.assertEqual(error, "ValueError: missing sales column")

    def test_safe_execute_allows_next_builtin(self):
        result, error = prompt_pipeline.safe_execute("result = next(iter(df['sales']))", self.df)

        self.assertIsNone(error)
        self.assertEqual(result, 100)

    def test_generate_ratings_analysis_code_uses_gemini_prompt_helper(self):
        ratings_df = pd.DataFrame(
            {
                "product": ["A", "B"],
                "rating": [4.5, 3.8],
            }
        )

        with mock.patch.object(prompt_pipeline, "_generate_gemini_content", return_value="result = 'ok'") as gemini_mock:
            result = prompt_pipeline.generate_ratings_analysis_code(ratings_df)

        self.assertEqual(result, "result = 'ok'")
        self.assertFalse(gemini_mock.call_args.kwargs["use_search"])
        prompt_text = gemini_mock.call_args.kwargs["prompt"]
        self.assertIn("product", prompt_text)
        self.assertIn("rating", prompt_text)
        self.assertIn("analyzes product or service ratings", prompt_text)
        self.assertIn("Calculate the rating distribution", prompt_text)
        self.assertIn("top-rated items, worst-rated items, and rating distribution", prompt_text)

    def test_generate_forecast_prep_code_uses_gemini_prompt_helper(self):
        forecast_df = pd.DataFrame(
            {
                "order_date": ["2024-01-01", "2024-01-02"],
                "sales": [120, 180],
            }
        )

        with mock.patch.object(prompt_pipeline, "_generate_gemini_content", return_value="result = 'forecast_ready'") as gemini_mock:
            result = prompt_pipeline.generate_forecast_prep_code(forecast_df)

        self.assertEqual(result, "result = 'forecast_ready'")
        self.assertFalse(gemini_mock.call_args.kwargs["use_search"])
        prompt_text = gemini_mock.call_args.kwargs["prompt"]
        self.assertIn("order_date", prompt_text)
        self.assertIn("sales", prompt_text)
        self.assertIn("time-series forecasting expert", prompt_text)
        self.assertIn("prepares the already loaded dataframe named `df` for future value prediction", prompt_text)
        self.assertIn("Use time-based aggregation", prompt_text)
        self.assertIn("Handle missing dates", prompt_text)
        self.assertIn("Raise `ValueError` if no clear date column exists or no clear forecast metric can be identified safely", prompt_text)

    def test_generate_sales_prediction_code_uses_gemini_prompt_helper(self):
        sales_df = pd.DataFrame(
            {
                "order_date": ["2024-01-01", "2024-01-02"],
                "sales": [120, 180],
            }
        )

        with mock.patch.object(prompt_pipeline, "_generate_gemini_content", return_value="result = 'predictions_ready'") as gemini_mock:
            result = prompt_pipeline.generate_sales_prediction_code(sales_df)

        self.assertEqual(result, "result = 'predictions_ready'")
        self.assertFalse(gemini_mock.call_args.kwargs["use_search"])
        prompt_text = gemini_mock.call_args.kwargs["prompt"]
        self.assertIn("order_date", prompt_text)
        self.assertIn("sales", prompt_text)
        self.assertIn("Build a predictive model", prompt_text)
        self.assertIn("Detect target column", prompt_text)
        self.assertIn("Split data", prompt_text)
        self.assertIn("Train model", prompt_text)
        self.assertIn("Evaluate accuracy", prompt_text)
        self.assertIn("Store result", prompt_text)
        self.assertIn("Python code only", prompt_text)

    def test_validate_ratings_dataset_uses_gemini_prompt_helper(self):
        ratings_df = pd.DataFrame(
            {
                "product": ["A", "B"],
                "rating": [4.5, 3.8],
            }
        )

        with mock.patch.object(prompt_pipeline, "_generate_gemini_content", return_value="VALID\nLooks usable.") as gemini_mock:
            result = prompt_pipeline.validate_ratings_dataset(ratings_df)

        self.assertEqual(result, "VALID\nLooks usable.")
        self.assertFalse(gemini_mock.call_args.kwargs["use_search"])
        prompt_text = gemini_mock.call_args.kwargs["prompt"]
        self.assertIn("product", prompt_text)
        self.assertIn("rating", prompt_text)
        self.assertIn("validating dataset for ratings analysis", prompt_text)

    def test_detect_intent_routes_rating_queries(self):
        self.assertEqual(prompt_pipeline.detect_intent("show average rating"), "rating")

    def test_detect_intent_routes_prediction_queries(self):
        self.assertEqual(prompt_pipeline.detect_intent("predict sales for next quarter"), "forecast")

    def test_detect_intent_defaults_to_general_queries(self):
        self.assertEqual(prompt_pipeline.detect_intent("top customers by revenue"), "general")

    def test_detect_analysis_intent_keeps_uppercase_compatibility(self):
        self.assertEqual(prompt_pipeline.detect_analysis_intent("show average rating"), "RATINGS")

    def test_classify_analysis_intent_uses_contract_taxonomy(self):
        self.assertEqual(prompt_pipeline.classify_analysis_intent("clean missing values"), "data_cleaning")
        self.assertEqual(prompt_pipeline.classify_analysis_intent("plot sales by region"), "visualization")
        self.assertEqual(prompt_pipeline.classify_analysis_intent("predict next month sales"), "prediction")
        self.assertEqual(prompt_pipeline.classify_analysis_intent("compare sales by region"), "comparison")
        self.assertEqual(prompt_pipeline.classify_analysis_intent("why did sales drop"), "root_cause")
        self.assertEqual(prompt_pipeline.classify_analysis_intent("summarize customer behavior"), "analysis")

    def test_run_builder_pipeline_uses_ratings_module(self):
        with mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"):
            with mock.patch.object(prompt_pipeline, "log_step"):
                with mock.patch.object(prompt_pipeline, "validate_ratings_dataset", return_value="VALID\nRatings data looks usable.") as ratings_validation_mock:
                    with mock.patch.object(prompt_pipeline, "generate_ratings_analysis_code", return_value="result = {'ok': 1}") as ratings_code_mock:
                        with mock.patch.object(prompt_pipeline, "execute_code", return_value=({"ok": 1}, None)) as execute_mock:
                            with mock.patch.object(prompt_pipeline, "summarize_for_non_technical_user", return_value="summary") as summary_mock:
                                with mock.patch.object(prompt_pipeline, "generate_insights", return_value="insights") as insights_mock:
                                    result = prompt_pipeline.run_builder_pipeline(
                                        "show product ratings",
                                        self.df,
                                        workflow_context={"workflow_id": "wf-1"},
                                    )

        self.assertEqual(result["intent"], "rating")
        ratings_validation_mock.assert_called_once()
        ratings_code_mock.assert_called_once()
        execute_mock.assert_called_once_with("result = {'ok': 1}", self.df)
        self.assertEqual(result["test_status"], "PASSED")
        self.assertEqual(result["summary"], "summary")
        self.assertEqual(result["insights"], "insights")
        self.assertIn("analysis_contract", result)
        self.assertEqual(result["analysis_contract"]["result_summary"], "summary")
        self.assertEqual(result["analysis_contract"]["insights"], ["insights"])
        self.assertIn("data_quality", result["analysis_contract"])
        self.assertIn("model_quality", result["analysis_contract"])
        self.assertIn("risk", result["analysis_contract"])
        self.assertIn("decision_layer", result["analysis_contract"])
        self.assertTrue(result["analysis_contract"]["decision_layer"]["decisions"])
        self.assertIn("learning_insights", result["analysis_contract"]["decision_layer"])
        self.assertTrue(result["recommendations"])
        self.assertTrue(result["business_decisions"])
        self.assertRegex(result["confidence"], r"^\d+/10$")
        self.assertIn("data_score", result)
        self.assertIn("pipeline_trace", result)
        self.assertEqual(result["pipeline_trace"][0]["stage"], "user_query")
        self.assertEqual(result["pipeline_trace"][-1]["stage"], "memory_update")
        self.assertIn("decision_engine", [stage["stage"] for stage in result["pipeline_trace"]])
        self.assertIsNone(result["anomalies"])
        self.assertEqual(result["workflow_context"], {"workflow_id": "wf-1"})
        summary_mock.assert_called_once()
        self.assertEqual(insights_mock.call_args.args[0], "show product ratings")
        self.assertEqual(insights_mock.call_args.args[1], {"ok": 1})
        self.assertTrue(insights_mock.call_args.args[2].equals(self.df))

    def test_run_builder_pipeline_uses_forecast_module(self):
        prepared_df = pd.DataFrame(
            {
                "order_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "sales": [120, 180, 210],
            }
        )

        with mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"):
            with mock.patch.object(prompt_pipeline, "log_step"):
                with mock.patch.object(prompt_pipeline, "validate_forecast_dataset", return_value="VALID\nForecast data looks usable.") as forecast_validation_mock:
                    with mock.patch.object(prompt_pipeline, "generate_forecast_prep_code", return_value="result = prepared_df") as prep_code_mock:
                        with mock.patch.object(prompt_pipeline, "generate_sales_prediction_code", return_value="result = {'forecast': 1}") as forecast_code_mock:
                            with mock.patch.object(prompt_pipeline, "execute_code", side_effect=[(prepared_df, None), ({"forecast": 1}, None)]) as execute_mock:
                                with mock.patch.object(prompt_pipeline, "summarize_for_non_technical_user", return_value="summary"):
                                    with mock.patch.object(prompt_pipeline, "generate_insights", return_value="insights"):
                                        result = prompt_pipeline.run_builder_pipeline("predict next year sales", self.forecast_df)

        self.assertEqual(result["intent"], "forecast")
        forecast_validation_mock.assert_called_once()
        prep_code_mock.assert_called_once()
        forecast_code_mock.assert_called_once()
        self.assertTrue(forecast_code_mock.call_args.args[0].equals(prepared_df))
        self.assertEqual(execute_mock.call_count, 2)
        self.assertEqual(execute_mock.call_args_list[0].args, ("result = prepared_df", self.forecast_df))
        self.assertEqual(execute_mock.call_args_list[1].args, ("result = {'forecast': 1}", prepared_df))
        self.assertEqual(result["test_status"], "PASSED")
        self.assertIn("Forecast Preparation", result["generated_code"])
        self.assertIn("Sales Prediction", result["generated_code"])
        self.assertEqual(result["analysis_contract"]["intent"], "prediction")
        self.assertEqual(result["analysis_contract"]["tool_used"], "PYTHON")
        self.assertEqual(result["analysis_contract"]["analysis_mode"], "prediction")
        self.assertEqual(result["analysis_contract"]["result_summary"], "summary")
        self.assertEqual(result["analysis_contract"]["insights"], ["insights"])
        self.assertIn("data_quality", result["analysis_contract"])
        self.assertIn("model_quality", result["analysis_contract"])
        self.assertIn("risk", result["analysis_contract"])
        self.assertIn("decision_layer", result["analysis_contract"])
        self.assertTrue(result["analysis_contract"]["decision_layer"]["decisions"])
        self.assertIn("learning_insights", result["analysis_contract"]["decision_layer"])
        self.assertTrue(result["recommendations"])
        self.assertTrue(result["business_decisions"])
        self.assertIn("data_score", result)
        self.assertIn("pipeline_trace", result)
        self.assertIn("decision_engine", [stage["stage"] for stage in result["pipeline_trace"]])
        self.assertIsNone(result["anomalies"])

    def test_run_builder_pipeline_reroutes_non_time_forecast_queries_to_ml(self):
        tabular_df = pd.DataFrame(
            {
                "study_hours": [2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8],
                "attendance": [70, 72, 75, 78, 80, 84, 74, 76, 79, 83, 86, 90],
                "final_score": [55, 58, 62, 67, 71, 76, 60, 64, 69, 73, 79, 85],
                "segment": ["A", "A", "B", "B", "C", "C", "A", "B", "B", "C", "C", "A"],
            }
        )

        prediction_result = pd.DataFrame(
            {
                "prediction_step": [1, 2, 3],
                "prediction_target": ["final_score", "final_score", "final_score"],
                "predicted_value": [87.0, 89.5, 92.0],
            }
        )

        with mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"):
            with mock.patch.object(prompt_pipeline, "log_step"):
                with mock.patch.object(prompt_pipeline, "validate_forecast_dataset") as forecast_validation_mock:
                    with mock.patch.object(prompt_pipeline, "execute_code", return_value=(prediction_result, None)) as execute_mock:
                        with mock.patch.object(prompt_pipeline, "summarize_for_non_technical_user", return_value="summary"):
                            with mock.patch.object(prompt_pipeline, "generate_insights", return_value="insights"):
                                result = prompt_pipeline.run_builder_pipeline("predict future final_score", tabular_df)

        forecast_validation_mock.assert_not_called()
        execute_mock.assert_called_once()
        self.assertEqual(result["analysis_contract"]["system_decision"]["selected_mode"], "ml")
        self.assertIn("no valid time column", result["analysis_contract"]["system_decision"]["reason"].lower())
        self.assertEqual(result["analysis_contract"]["analysis_mode"], "prediction")
        self.assertEqual(result["test_status"], "PASSED")

    def test_run_builder_pipeline_stops_on_invalid_module_validation(self):
        forecast_df = self.forecast_df.copy()
        with mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"):
            with mock.patch.object(prompt_pipeline, "validate_forecast_dataset", return_value="INVALID\nMissing sales column."):
                result = prompt_pipeline.run_builder_pipeline("predict next year sales", forecast_df)

        self.assertEqual(result["intent"], "forecast")
        self.assertEqual(result["test_status"], "PASSED")
        self.assertEqual(result["analysis_contract"]["system_decision"]["selected_mode"], "ml")
        self.assertIn("switching to predictive modeling", result["analysis_contract"]["system_decision"]["suggestion"].lower())
        self.assertIsNone(result["error"])

    def test_run_builder_pipeline_stops_when_forecast_prep_is_not_tabular(self):
        forecast_df = self.forecast_df.copy()
        with mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"):
            with mock.patch.object(prompt_pipeline, "log_step"):
                with mock.patch.object(prompt_pipeline, "validate_forecast_dataset", return_value="VALID\nForecast data looks usable."):
                    with mock.patch.object(prompt_pipeline, "generate_forecast_prep_code", return_value="result = 'not a dataframe'"):
                        with mock.patch.object(prompt_pipeline, "generate_sales_prediction_code") as forecast_code_mock:
                            with mock.patch.object(prompt_pipeline, "execute_code", return_value=("not a dataframe", None)):
                                result = prompt_pipeline.run_builder_pipeline("forecast next month sales", forecast_df)

        self.assertEqual(result["intent"], "forecast")
        self.assertEqual(result["test_status"], "PASSED")
        self.assertEqual(result["analysis_contract"]["system_decision"]["selected_mode"], "ml")
        self.assertIn("forecasting could not run safely", result["analysis_contract"]["system_decision"]["reason"].lower())
        self.assertIsNone(result["error"])
        forecast_code_mock.assert_not_called()

    def test_run_builder_pipeline_keeps_general_fallback(self):
        with mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"):
            with mock.patch.object(prompt_pipeline, "log_step"):
                with mock.patch.object(prompt_pipeline, "generate_code", return_value="result = {'rows': 2}") as generate_code_mock:
                    with mock.patch.object(prompt_pipeline, "execute_code", return_value=({"rows": 2}, None)) as execute_mock:
                        with mock.patch.object(prompt_pipeline, "summarize_for_non_technical_user", return_value="summary"):
                            with mock.patch.object(prompt_pipeline, "generate_insights", return_value="insights"):
                                result = prompt_pipeline.run_builder_pipeline("model customer behavior patterns", self.df)

        self.assertEqual(result["intent"], "general")
        self.assertEqual(result["module_validation"], "VALID\nGeneral analysis does not require specialized module validation.")
        generate_code_mock.assert_called_once()
        execute_mock.assert_called_once_with("result = {'rows': 2}", self.df)
        self.assertEqual(result["test_status"], "PASSED")
        self.assertEqual(result["summary"], "summary")
        self.assertEqual(result["insights"], "insights")
        self.assertEqual(result["analysis_contract"]["intent"], "analysis")
        self.assertEqual(result["analysis_contract"]["tool_used"], "PYTHON")
        self.assertEqual(result["analysis_contract"]["result_summary"], "summary")
        self.assertEqual(result["analysis_contract"]["insights"], ["insights"])
        self.assertTrue(result["recommendations"])
        self.assertTrue(result["business_decisions"])
        self.assertEqual(result["pipeline_trace"][1]["stage"], "intent_detection")
        self.assertEqual(result["pipeline_trace"][4]["stage"], "validation_data_score")
        self.assertIsNone(result["anomalies"])

    def test_run_builder_pipeline_retries_and_records_fixed_code_after_execution_error(self):
        with mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"):
            with mock.patch.object(prompt_pipeline, "log_step"):
                with mock.patch.object(prompt_pipeline, "generate_code", return_value="raise ValueError('missing revenue')"):
                    with mock.patch.object(prompt_pipeline, "fix_code", return_value="result = {'region': 'North', 'sales': 100}") as fix_code_mock:
                        with mock.patch.object(
                            prompt_pipeline,
                            "execute_code",
                            side_effect=[
                                (None, "ValueError: missing revenue"),
                                ({"region": "North", "sales": 100}, None),
                            ],
                        ) as execute_mock:
                            with mock.patch.object(prompt_pipeline, "summarize_for_non_technical_user", return_value="summary"):
                                with mock.patch.object(prompt_pipeline, "generate_insights", return_value="insights"):
                                    result = prompt_pipeline.run_builder_pipeline("Model customer behavior patterns", self.df)

        self.assertEqual(result["test_status"], "PASSED")
        self.assertTrue(result["fix_applied"])
        self.assertEqual(result["generated_code"], "raise ValueError('missing revenue')")
        self.assertEqual(result["fixed_code"], "result = {'region': 'North', 'sales': 100}")
        self.assertEqual(result["code"], "result = {'region': 'North', 'sales': 100}")
        self.assertEqual(result["result"], {"region": "North", "sales": 100})
        self.assertIsNone(result["error"])
        self.assertIn("Automatic fix applied successfully", result["fix_status"])
        self.assertIn("analysis_contract", result)
        self.assertRegex(result["confidence"], r"^\d+/10$")
        self.assertIn("data_score", result)
        self.assertEqual(execute_mock.call_count, 2)
        fix_code_mock.assert_called_once_with(
            "raise ValueError('missing revenue')",
            "ValueError: missing revenue",
            self.df,
            "Model customer behavior patterns",
            model=prompt_pipeline.DEFAULT_GEMINI_MODEL,
        )

    def test_run_builder_pipeline_uses_deterministic_average_by_shortcut(self):
        with mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"):
            with mock.patch.object(prompt_pipeline, "log_step"):
                with mock.patch.object(prompt_pipeline, "generate_code") as generate_code_mock:
                    with mock.patch.object(prompt_pipeline, "summarize_for_non_technical_user", return_value="summary"):
                        with mock.patch.object(prompt_pipeline, "generate_insights", return_value="insights"):
                            result = prompt_pipeline.run_builder_pipeline("Average sales by region", self.df)

        self.assertEqual(result["intent"], "general")
        self.assertEqual(result["test_status"], "PASSED")
        self.assertIn("Excel", result["build_plan"])
        self.assertIn("sales", result["generated_code"])
        self.assertIn("region", result["generated_code"])
        self.assertFalse(result["fix_applied"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["tool_used"], "EXCEL")
        self.assertEqual(result["analysis_contract"]["tool_used"], "EXCEL")
        self.assertIn("excel_analysis", result["analysis_contract"])
        self.assertEqual(list(result["result"].columns), ["region", "sales"])
        self.assertEqual(result["result"].iloc[0]["region"], "South")
        self.assertEqual(result["result"].iloc[0]["sales"], 200.0)
        self.assertEqual(result["analysis_contract"]["intent"], "analysis")
        self.assertTrue(result["recommendations"])
        self.assertEqual(result["pipeline_trace"][3]["stage"], "forecast_ml")
        generate_code_mock.assert_not_called()

    def test_run_builder_pipeline_blocks_weak_prediction_inputs_in_contract(self):
        prediction_df = pd.DataFrame({"sales": [10, 12, 15]})

        with mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"):
            result = prompt_pipeline.run_builder_pipeline("predict sales", prediction_df)

        self.assertEqual(result["test_status"], "PASSED")
        self.assertEqual(result["analysis_contract"]["intent"], "prediction")
        self.assertEqual(result["analysis_contract"]["tool_used"], "EXCEL")
        self.assertEqual(result["analysis_contract"]["analysis_mode"], "prediction")
        self.assertEqual(result["analysis_contract"]["model_quality"], "not_applicable")
        self.assertTrue(result["warnings"])
        self.assertIsNone(result["error"])
        self.assertLess(result["data_score"]["score"], 90)
        excel_step = self._find_plan_step(result["analysis_contract"]["execution_plan"], "EXCEL")
        self.assertIsNotNone(excel_step)
        self.assertTrue(excel_step.get("fallback_reason"))

    def test_run_builder_pipeline_executes_multi_tool_orchestration(self):
        multi_df = pd.DataFrame(
            {
                "order_date": pd.date_range("2024-01-01", periods=14, freq="D"),
                "customer": ["A", "B", "C", "A", "B", "C", "D", "E", "A", "B", "C", "D", "E", "F"],
                "sales": [100, 120, 110, 140, 160, 150, 170, 180, 190, 200, 210, 220, 230, 240],
            }
        )

        def python_side_effect(code, frame):
            if "LinearRegression" in code:
                return (
                    pd.DataFrame(
                        {
                            "order_date": pd.date_range("2024-02-01", periods=3, freq="D"),
                            "sales": [170.0, 176.0, 182.0],
                        }
                    ),
                    None,
                )
            return (None, "Unexpected Python code")

        with mock.patch.object(prompt_pipeline, "_ensure_builder_configuration"):
            with mock.patch.object(prompt_pipeline, "log_step"):
                with mock.patch.object(prompt_pipeline, "summarize_for_non_technical_user", return_value="summary"):
                    with mock.patch.object(prompt_pipeline, "generate_insights", return_value="insights"):
                        with mock.patch.object(prompt_pipeline, "execute_code", side_effect=python_side_effect):
                            result = prompt_pipeline.run_builder_pipeline(
                                "Top customers + prediction + dashboard",
                                multi_df,
                            )

        self.assertEqual(result["test_status"], "PASSED")
        self.assertEqual([step["tool"] for step in result["execution_plan"]], ["SQL", "PYTHON", "BI"])
        self.assertTrue(result["execution_trace"])
        self.assertTrue(any(step["status"] == "completed" for step in result["execution_trace"]))
        self.assertEqual(result["analysis_contract"]["tool_used"], "PYTHON")
        self.assertEqual(result["analysis_contract"]["analysis_mode"], "prediction")
        self.assertTrue(result["dashboard"])
        self.assertTrue(result["dashboard"]["charts"])
        self.assertIn("optimization", result)
        self.assertTrue(result["optimization"]["optimized"])

    def test_validate_forecast_dataset_uses_gemini_prompt_helper(self):
        forecast_df = pd.DataFrame(
            {
                "order_date": ["2024-01-01", "2024-01-02"],
                "sales": [120, 180],
            }
        )

        with mock.patch.object(prompt_pipeline, "_generate_gemini_content", return_value="VALID\nLooks usable.") as gemini_mock:
            result = prompt_pipeline.validate_forecast_dataset(forecast_df)

        self.assertEqual(result, "VALID\nLooks usable.")
        self.assertFalse(gemini_mock.call_args.kwargs["use_search"])
        prompt_text = gemini_mock.call_args.kwargs["prompt"]
        self.assertIn("order_date", prompt_text)
        self.assertIn("sales", prompt_text)
        self.assertIn("validating dataset for forecasting", prompt_text)
        self.assertIn("First line must be exactly `VALID` or `INVALID`", prompt_text)


if __name__ == "__main__":
    unittest.main()
