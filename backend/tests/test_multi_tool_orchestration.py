import unittest

import pandas as pd

from backend.services.execution_engine import execute_plan
from backend.services.tool_planner import build_execution_plan


class MultiToolOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.sales_df = pd.DataFrame(
            {
                "order_date": pd.date_range("2025-01-01", periods=8, freq="D"),
                "region": ["North", "South", "North", "West", "South", "North", "West", "South"],
                "customer": ["A", "B", "A", "C", "B", "D", "C", "D"],
                "sales": [100, 140, 125, 90, 160, 180, 95, 170],
                "revenue": [1000, 1400, 1250, 900, 1600, 1800, 950, 1700],
            }
        )
        self.base_preflight = {"warnings": [], "blocking_errors": []}

    def test_build_execution_plan_chains_sql_then_python_for_top_customers_prediction(self):
        analysis_plan = {
            "intent": "prediction",
            "analysis_type": "aggregation",
            "metric_column": "revenue",
            "group_column": "customer",
            "datetime_column": "order_date",
            "required_columns": ["customer", "revenue", "order_date"],
        }

        execution_plan = build_execution_plan(
            "Top customers + prediction",
            self.sales_df,
            plan=analysis_plan,
            preflight=self.base_preflight,
        )

        self.assertEqual([step["tool"] for step in execution_plan], ["SQL", "PYTHON"])

    def test_build_execution_plan_chains_python_then_bi_for_sales_trends_dashboard(self):
        analysis_plan = {
            "intent": "visualization",
            "analysis_type": "aggregation",
            "metric_column": "sales",
            "group_column": "region",
            "datetime_column": "order_date",
            "required_columns": ["order_date", "sales", "region"],
        }

        execution_plan = build_execution_plan(
            "Sales trends + dashboard",
            self.sales_df,
            plan=analysis_plan,
            preflight=self.base_preflight,
        )

        self.assertEqual([step["tool"] for step in execution_plan], ["PYTHON", "BI"])

    def test_build_execution_plan_chains_excel_python_bi_for_clean_analyze_visualize(self):
        analysis_plan = {
            "intent": "data_cleaning",
            "analysis_type": "aggregation",
            "metric_column": "sales",
            "group_column": "region",
            "datetime_column": "order_date",
            "required_columns": ["order_date", "sales", "region"],
        }

        execution_plan = build_execution_plan(
            "Clean + analyze + visualize",
            self.sales_df,
            plan=analysis_plan,
            preflight=self.base_preflight,
        )

        self.assertEqual([step["tool"] for step in execution_plan], ["EXCEL", "PYTHON", "BI"])

    def test_execute_plan_passes_context_from_sql_to_python_and_python_to_bi(self):
        analysis_plan = {
            "intent": "prediction",
            "analysis_type": "aggregation",
            "metric_column": "revenue",
            "group_column": "customer",
            "datetime_column": "order_date",
            "required_columns": ["customer", "revenue", "order_date"],
        }

        plan = [
            {
                "step": 1,
                "tool": "SQL",
                "task": "Get the top customers using SQL-style grouping and ranking.",
                "query": "Top customers by revenue",
            },
            {
                "step": 2,
                "tool": "PYTHON",
                "task": "Run Python prediction on the current context.",
                "query": "Predict future revenue",
            },
            {
                "step": 3,
                "tool": "BI",
                "task": "Build a dashboard from the current context.",
                "query": "Build dashboard",
            },
        ]

        python_inputs = {}

        def python_runner(**kwargs):
            frame = kwargs["df"]
            python_inputs["columns"] = list(frame.columns)
            self.assertIn("customer", frame.columns)
            self.assertTrue(any(column.endswith("revenue") for column in frame.columns if column != "customer"))
            return {
                "result": pd.DataFrame(
                    {
                        "order_date": pd.date_range("2025-02-01", periods=3, freq="D"),
                        "revenue": [1820.0, 1885.0, 1940.0],
                    }
                ),
                "generated_code": "result = forecast_df",
                "final_code": "result = forecast_df",
                "error": None,
                "fix_applied": False,
                "fix_status": "Execution passed without needing an automatic fix.",
                "fixed_code": None,
                "analysis_method": "test_python_runner",
                "module_validation": "VALID\nPython step selected.",
                "python_steps": ["prepare context", "predict values"],
                "warnings": [],
            }

        payload = execute_plan(
            plan,
            self.sales_df,
            query="Top customers + prediction + dashboard",
            analysis_plan=analysis_plan,
            preflight=self.base_preflight,
            tables={"df": self.sales_df},
            python_runner=python_runner,
        )

        self.assertEqual(python_inputs["columns"][0], "customer")
        self.assertEqual([entry["status"] for entry in payload["execution_trace"]], ["completed", "completed", "completed"])
        self.assertTrue(all("execution_time_ms" in entry for entry in payload["execution_trace"]))
        self.assertTrue(all("cost_estimate" in entry for entry in payload["execution_trace"]))
        self.assertTrue(payload["dashboard"]["charts"])
        self.assertEqual(payload["execution_plan"][1]["tool"], "PYTHON")
        self.assertIn("optimization", payload)

    def test_execute_plan_falls_back_to_excel_when_python_step_fails(self):
        analysis_plan = {
            "intent": "analysis",
            "analysis_type": "aggregation",
            "metric_column": "sales",
            "group_column": "region",
            "required_columns": ["region", "sales"],
        }

        plan = [
            {
                "step": 1,
                "tool": "PYTHON",
                "task": "Run Python analysis on the current context.",
                "query": "Analyze sales",
            }
        ]

        payload = execute_plan(
            plan,
            self.sales_df,
            query="Analyze sales",
            analysis_plan=analysis_plan,
            preflight=self.base_preflight,
            tables={"df": self.sales_df},
            python_runner=lambda **kwargs: {
                "result": None,
                "generated_code": "raise RuntimeError('boom')",
                "final_code": "raise RuntimeError('boom')",
                "error": "RuntimeError: boom",
                "fix_applied": False,
                "fix_status": "Execution failed before an automatic fix could recover the analysis.",
                "fixed_code": None,
                "analysis_method": "test_python_runner",
                "module_validation": "VALID\nPython step selected.",
                "python_steps": ["attempt analysis"],
                "warnings": [],
            },
        )

        self.assertIsNone(payload["error"])
        self.assertTrue(any(entry["status"] == "fallback_completed" for entry in payload["execution_trace"]))
        self.assertIsNotNone(payload["excel_analysis"])
        self.assertIn("optimization", payload)


if __name__ == "__main__":
    unittest.main()
