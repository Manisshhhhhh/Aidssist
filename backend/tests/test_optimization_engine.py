import time
import unittest
from unittest import mock

import pandas as pd

from backend.services.execution_engine import execute_plan
from backend.services.optimizer import find_parallel_steps, optimize_plan


class OptimizationEngineTests(unittest.TestCase):
    def setUp(self):
        self.sales_df = pd.DataFrame(
            {
                "order_date": pd.date_range("2025-01-01", periods=12, freq="D"),
                "region": ["North", "South", "North", "West", "South", "North", "West", "South", "North", "West", "South", "North"],
                "customer": ["A", "B", "A", "C", "B", "D", "C", "D", "A", "C", "B", "D"],
                "sales": [100, 140, 125, 90, 160, 180, 95, 170, 190, 110, 175, 205],
                "revenue": [1000, 1400, 1250, 900, 1600, 1800, 950, 1700, 1900, 1100, 1750, 2050],
            }
        )
        self.base_preflight = {"warnings": [], "blocking_errors": []}

    def test_optimize_plan_removes_redundant_steps_and_prefers_cheaper_summary_tool(self):
        plan = [
            {
                "step": 1,
                "tool": "SQL",
                "task": "Show sales by region",
                "query": "Show sales by region",
            },
            {
                "step": 2,
                "tool": "SQL",
                "task": "Show sales by region",
                "query": "Show sales by region",
            },
        ]

        with mock.patch(
            "backend.services.optimizer.get_average_tool_time",
            side_effect=lambda tool: {"SQL": 60.0, "EXCEL": 10.0, "PYTHON": 120.0, "BI": 80.0}.get(tool, 100.0),
        ):
            optimized_plan = optimize_plan(plan, df=self.sales_df, preflight=self.base_preflight)

        self.assertEqual(len(optimized_plan), 1)
        self.assertEqual(optimized_plan[0]["tool"], "EXCEL")
        self.assertEqual(optimized_plan[0]["cost_estimate"], "low")

    def test_find_parallel_steps_identifies_independent_terminal_steps(self):
        plan = [
            {
                "step": 1,
                "tool": "SQL",
                "task": "Get top customers",
                "query": "Top customers",
                "depends_on": [],
                "uses_context": False,
            },
            {
                "step": 2,
                "tool": "EXCEL",
                "task": "Show sales by region",
                "query": "Show sales by region",
                "depends_on": [],
                "uses_context": False,
            },
        ]

        self.assertEqual(find_parallel_steps(plan), [[1, 2]])

    def test_execute_plan_tracks_cost_and_performance(self):
        plan = [
            {
                "step": 1,
                "tool": "PYTHON",
                "task": "Run Python analysis",
                "query": "Analyze sales",
            }
        ]

        payload = execute_plan(
            plan,
            self.sales_df,
            query="Analyze sales",
            analysis_plan={"intent": "analysis", "metric_column": "sales", "group_column": "region"},
            preflight=self.base_preflight,
            tables={"df": self.sales_df},
            python_runner=lambda **kwargs: {
                "result": pd.DataFrame({"sales": [999.0]}),
                "generated_code": "result = df[['sales']].head(1)",
                "final_code": "result = df[['sales']].head(1)",
                "error": None,
                "fix_applied": False,
                "fix_status": "Execution passed without needing an automatic fix.",
                "fixed_code": None,
                "analysis_method": "test_python_runner",
                "module_validation": "VALID\nPython step selected.",
                "python_steps": ["execute analysis"],
                "warnings": [],
            },
            optimize=False,
            enable_parallel=False,
        )

        self.assertTrue(payload["execution_trace"])
        self.assertGreaterEqual(payload["execution_trace"][0]["execution_time_ms"], 0)
        self.assertEqual(payload["execution_trace"][0]["cost_estimate"], "low")
        self.assertIn("optimization", payload)
        self.assertGreaterEqual(payload["optimization"]["execution_time_total"], 0)

    def test_optimized_execution_is_faster_without_losing_correctness(self):
        duplicate_plan = [
            {
                "step": 1,
                "tool": "PYTHON",
                "task": "Run Python analysis",
                "query": "Analyze sales",
            },
            {
                "step": 2,
                "tool": "PYTHON",
                "task": "Run Python analysis",
                "query": "Analyze sales",
            },
        ]

        def slow_python_runner(**kwargs):
            time.sleep(0.04)
            return {
                "result": pd.DataFrame({"sales": [999.0]}),
                "generated_code": "result = df[['sales']].head(1)",
                "final_code": "result = df[['sales']].head(1)",
                "error": None,
                "fix_applied": False,
                "fix_status": "Execution passed without needing an automatic fix.",
                "fixed_code": None,
                "analysis_method": "test_python_runner",
                "module_validation": "VALID\nPython step selected.",
                "python_steps": ["execute analysis"],
                "warnings": [],
            }

        unoptimized = execute_plan(
            duplicate_plan,
            self.sales_df,
            query="Analyze sales",
            analysis_plan={"intent": "analysis", "metric_column": "sales", "group_column": "region"},
            preflight=self.base_preflight,
            tables={"df": self.sales_df},
            python_runner=slow_python_runner,
            optimize=False,
            enable_parallel=False,
        )
        optimized = execute_plan(
            duplicate_plan,
            self.sales_df,
            query="Analyze sales",
            analysis_plan={"intent": "analysis", "metric_column": "sales", "group_column": "region"},
            preflight=self.base_preflight,
            tables={"df": self.sales_df},
            python_runner=slow_python_runner,
            optimize=True,
            enable_parallel=False,
        )

        self.assertEqual(len(unoptimized["execution_plan"]), 2)
        self.assertEqual(len(optimized["execution_plan"]), 1)
        self.assertTrue(optimized["optimization"]["optimized"])
        self.assertLess(optimized["optimization"]["execution_time_total"], unoptimized["optimization"]["execution_time_total"])
        self.assertTrue(unoptimized["result"].equals(optimized["result"]))

    def test_parallel_execution_runs_independent_steps_faster(self):
        plan = [
            {
                "step": 1,
                "tool": "SQL",
                "task": "Get top customers",
                "query": "Top customers",
                "depends_on": [],
                "uses_context": False,
            },
            {
                "step": 2,
                "tool": "EXCEL",
                "task": "Show sales by region",
                "query": "Show sales by region",
                "depends_on": [],
                "uses_context": False,
            },
        ]

        def slow_sql(*args, **kwargs):
            time.sleep(0.05)
            return {
                "result": pd.DataFrame({"customer": ["A"], "sum_revenue": [5900.0]}),
                "sql_plan": "SELECT customer, SUM(revenue) FROM df GROUP BY customer",
                "warnings": [],
                "unsupported": False,
            }

        def slow_excel(*args, **kwargs):
            time.sleep(0.05)
            return {
                "result": pd.DataFrame({"region": ["North"], "sales": [620.0]}),
                "excel_analysis": {
                    "pivot_table": {"index": ["region"], "values": ["sales"], "aggfunc": "sum"},
                    "aggregations": {"sales_sum": 1740.0},
                    "summary": {"group_column": "region", "metric_column": "sales"},
                },
                "warnings": [],
            }

        with mock.patch("backend.services.execution_engine.run_sql_analysis", side_effect=slow_sql):
            with mock.patch("backend.services.execution_engine.run_excel_analysis", side_effect=slow_excel):
                serial_start = time.perf_counter()
                serial_payload = execute_plan(
                    plan,
                    self.sales_df,
                    query="Top customers and show sales by region",
                    analysis_plan={"intent": "analysis", "metric_column": "sales", "group_column": "region"},
                    preflight=self.base_preflight,
                    tables={"df": self.sales_df},
                    optimize=False,
                    enable_parallel=False,
                )
                serial_elapsed = time.perf_counter() - serial_start

                parallel_start = time.perf_counter()
                parallel_payload = execute_plan(
                    plan,
                    self.sales_df,
                    query="Top customers and show sales by region",
                    analysis_plan={"intent": "analysis", "metric_column": "sales", "group_column": "region"},
                    preflight=self.base_preflight,
                    tables={"df": self.sales_df},
                    optimize=False,
                    enable_parallel=True,
                )
                parallel_elapsed = time.perf_counter() - parallel_start

        self.assertFalse(serial_payload["optimization"]["parallel_execution"])
        self.assertTrue(parallel_payload["optimization"]["parallel_execution"])
        self.assertLess(parallel_elapsed, serial_elapsed)


if __name__ == "__main__":
    unittest.main()
