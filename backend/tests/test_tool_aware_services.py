import unittest

import pandas as pd

from backend.services.dashboard_engine import build_dashboard_output
from backend.services.excel_engine import run_excel_analysis
from backend.services.sql_engine import run_sql_analysis
from backend.services.tool_selector import select_tool


class ToolAwareServicesTests(unittest.TestCase):
    def setUp(self):
        self.sales_df = pd.DataFrame(
            {
                "order_date": pd.date_range("2025-01-01", periods=6, freq="D"),
                "region": ["North", "South", "North", "West", "South", "North"],
                "customer": ["A", "B", "A", "C", "B", "D"],
                "sales": [100, 140, 125, 90, 160, 180],
                "revenue": [1000, 1400, 1250, 900, 1600, 1800],
            }
        )
        self.base_preflight = {"warnings": [], "blocking_errors": []}

    def test_select_tool_prefers_excel_for_simple_summary(self):
        plan = {
            "intent": "analysis",
            "analysis_type": "aggregation",
            "metric_column": "sales",
            "group_column": "region",
            "required_columns": ["sales", "region"],
        }

        selection = select_tool("Show sales by region", self.sales_df, plan=plan, preflight=self.base_preflight)

        self.assertEqual(selection["tool_used"], "EXCEL")
        self.assertEqual(selection["analysis_mode"], "ad-hoc")

    def test_select_tool_prefers_sql_for_ranked_lookup(self):
        plan = {
            "intent": "comparison",
            "analysis_type": "aggregation",
            "metric_column": "revenue",
            "group_column": "customer",
            "required_columns": ["customer", "revenue"],
        }

        selection = select_tool("Get top customers", self.sales_df, plan=plan, preflight=self.base_preflight)

        self.assertEqual(selection["tool_used"], "SQL")

    def test_select_tool_prefers_python_for_prediction(self):
        plan = {
            "intent": "prediction",
            "analysis_type": "time_series",
            "metric_column": "revenue",
            "target_column": "revenue",
            "datetime_column": "order_date",
            "required_columns": ["order_date", "revenue"],
        }

        selection = select_tool("Predict next month revenue", self.sales_df, plan=plan, preflight=self.base_preflight)

        self.assertEqual(selection["tool_used"], "PYTHON")
        self.assertEqual(selection["analysis_mode"], "prediction")

    def test_select_tool_prefers_bi_for_dashboard_requests(self):
        plan = {
            "intent": "visualization",
            "analysis_type": "aggregation",
            "metric_column": "sales",
            "group_column": "region",
            "datetime_column": "order_date",
            "required_columns": ["sales", "region"],
        }

        selection = select_tool("Build dashboard", self.sales_df, plan=plan, preflight=self.base_preflight)

        self.assertEqual(selection["tool_used"], "BI")
        self.assertEqual(selection["analysis_mode"], "dashboard")

    def test_select_tool_falls_back_to_excel_for_weak_prediction_inputs(self):
        weak_df = pd.DataFrame({"sales": [10, 12, 15]})
        plan = {
            "intent": "prediction",
            "analysis_type": "ml",
            "metric_column": "sales",
            "target_column": "sales",
            "required_columns": ["sales"],
        }
        preflight = {"warnings": [], "blocking_errors": ["Prediction needs more rows."]}

        selection = select_tool("Predict sales", weak_df, plan=plan, preflight=preflight)

        self.assertEqual(selection["tool_used"], "EXCEL")
        self.assertEqual(selection["analysis_mode"], "prediction")
        self.assertTrue(selection["fallback_reason"])

    def test_run_sql_analysis_returns_sql_plan(self):
        plan = {
            "metric_column": "revenue",
            "group_column": "customer",
        }

        payload = run_sql_analysis(
            "Get top customers",
            self.sales_df,
            tables={"df": self.sales_df},
            plan=plan,
            preflight=self.base_preflight,
        )

        self.assertFalse(payload["unsupported"])
        self.assertIn("SELECT customer", payload["sql_plan"])
        self.assertFalse(payload["result"].empty)

    def test_run_sql_analysis_flags_join_without_multiple_tables(self):
        plan = {
            "metric_column": "sales",
            "group_column": "region",
        }

        payload = run_sql_analysis(
            "Join orders and customers",
            self.sales_df,
            tables={"df": self.sales_df},
            plan=plan,
            preflight=self.base_preflight,
        )

        self.assertTrue(payload["unsupported"])
        self.assertTrue(payload["warnings"])

    def test_run_excel_analysis_returns_pivot_and_aggregations(self):
        plan = {
            "metric_column": "sales",
            "group_column": "region",
        }

        payload = run_excel_analysis(
            "Show sales by region",
            self.sales_df,
            plan=plan,
            preflight=self.base_preflight,
        )

        self.assertIn("pivot_table", payload["excel_analysis"])
        self.assertIn("aggregations", payload["excel_analysis"])
        self.assertFalse(payload["result"].empty)

    def test_build_dashboard_output_returns_charts_and_kpis(self):
        plan = {
            "metric_column": "sales",
            "group_column": "region",
            "datetime_column": "order_date",
        }

        payload = build_dashboard_output(
            "Build dashboard",
            self.sales_df,
            plan=plan,
            preflight=self.base_preflight,
        )

        self.assertTrue(payload["dashboard"]["charts"])
        self.assertTrue(payload["dashboard"]["kpis"])
        self.assertTrue(payload["dashboard"]["filters"])
        self.assertTrue(payload["dashboard"]["drilldown_ready"])
        self.assertEqual(payload["dashboard"]["visualization_type"], "line")
        self.assertIn("context", payload)
        self.assertIn("suggestions", payload)
        self.assertTrue(payload["recommended_next_step"])
        self.assertTrue(payload["suggested_questions"])


if __name__ == "__main__":
    unittest.main()
