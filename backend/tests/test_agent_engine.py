from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

import backend.services.agent_engine as agent_engine
from backend.services.agent_engine import run_analysis_agent


class AgentEngineTests(unittest.TestCase):
    def setUp(self):
        agent_engine.analysis_memory.clear()
        self.df = pd.DataFrame(
            {
                "country": ["India", "India", "US", "US", "Brazil", "Japan"],
                "region": ["Asia", "Asia", "North America", "North America", "South America", "Asia"],
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]
                ),
                "confirmed_cases": [120, 130, 200, 150, 90, 80],
                "deaths": [5, 7, 10, 6, 4, 2],
            }
        )

    def _print_response(self, query: str, response: dict) -> None:
        print(f"\nQUERY: {query}")
        print("generated code:")
        print(response["code"])
        print("result:")
        print(response["result"])
        print("insight:")
        print(response["insight"])
        print("suggestions:")
        print(response["suggestions"])
        print("error:")
        print(response["error"])

    def test_top_5_countries_by_confirmed_cases(self):
        query = "top 5 countries by confirmed cases"
        llm_code = (
            "result = (\n"
            "    df.groupby('country', dropna=False)['confirmed_cases']\n"
            "    .sum()\n"
            "    .sort_values(ascending=False)\n"
            "    .head(5)\n"
            "    .reset_index()\n"
            ")"
        )
        with mock.patch("backend.services.agent_engine.generate_code_with_llm", return_value=llm_code):
            response = run_analysis_agent(self.df, query)
        self._print_response(query, response)

        self.assertIsNone(response["error"])
        self.assertIsInstance(response["result"], pd.DataFrame)
        self.assertEqual(list(response["result"]["country"]), ["US", "India", "Brazil", "Japan"])
        self.assertEqual(list(response["result"]["confirmed_cases"]), [350, 250, 90, 80])
        self.assertIn("suggestions", response)
        self.assertTrue(response["suggestions"])

    def test_average_deaths(self):
        query = "average deaths"
        with mock.patch("backend.services.agent_engine.generate_code_with_llm", side_effect=RuntimeError("Gemini API error")):
            response = run_analysis_agent(self.df, query)
        self._print_response(query, response)

        self.assertIsNone(response["error"])
        self.assertAlmostEqual(response["result"], 34 / 6, places=6)
        self.assertEqual(response.get("llm_fallback_reason"), "Gemini API error")
        self.assertTrue(response["suggestions"])

    def test_group_by_country(self):
        query = "group by country"
        llm_code = (
            "result = (\n"
            "    df.groupby('country', dropna=False)\n"
            "    .sum(numeric_only=True)\n"
            "    .reset_index()\n"
            ")"
        )
        with mock.patch("backend.services.agent_engine.generate_code_with_llm", return_value=llm_code):
            response = run_analysis_agent(self.df, query)
        self._print_response(query, response)

        self.assertIsNone(response["error"])
        self.assertIsInstance(response["result"], pd.DataFrame)
        self.assertIn("confirmed_cases", response["result"].columns)
        self.assertIn("deaths", response["result"].columns)
        india_row = response["result"].loc[response["result"]["country"] == "India"].iloc[0]
        self.assertEqual(india_row["confirmed_cases"], 250)
        self.assertEqual(india_row["deaths"], 12)
        self.assertTrue(response["suggestions"])

    def test_llm_execution_error_falls_back_to_rule_based_generator(self):
        query = "average deaths"
        broken_llm_code = "result = df['missing_column'].mean()"
        with mock.patch("backend.services.agent_engine.generate_code_with_llm", return_value=broken_llm_code):
            response = run_analysis_agent(self.df, query)
        self._print_response(query, response)

        self.assertIsNone(response["error"])
        self.assertAlmostEqual(response["result"], 34 / 6, places=6)
        self.assertEqual(response.get("llm_fallback_reason"), "'missing_column'")

    def test_trend_over_time_falls_back_when_llm_code_is_unsafe(self):
        query = "trend over time"
        unsafe_code = "import os\nresult = 1"
        with mock.patch("backend.services.agent_engine.generate_code_with_llm", return_value=unsafe_code):
            response = run_analysis_agent(self.df, query)
        self._print_response(query, response)

        self.assertIsNone(response["error"])
        self.assertIsInstance(response["result"], pd.DataFrame)
        self.assertEqual(list(response["result"]["date"]), list(pd.to_datetime(["2024-01-01", "2024-01-02"])))
        self.assertEqual(list(response["result"]["confirmed_cases"]), [410, 360])
        self.assertEqual(response.get("llm_fallback_reason"), "Unsafe code detected")

    def test_invalid_query(self):
        query = "invalid query"
        with mock.patch("backend.services.agent_engine.generate_code_with_llm", side_effect=RuntimeError("Gemini unavailable")):
            response = run_analysis_agent(self.df, query)
        self._print_response(query, response)

        self.assertIsNotNone(response["error"])
        self.assertEqual(response["result"], None)
        self.assertEqual(response["insight"], "")

    def test_memory_context_and_suggestions_change_across_queries(self):
        captured_contexts: list[list[dict[str, str]]] = []

        def _llm_side_effect(df_head, user_query, memory_context=None):
            captured_contexts.append(list(memory_context or []))
            if user_query == "top countries":
                return (
                    "result = (\n"
                    "    df['country']\n"
                    "    .value_counts(dropna=False)\n"
                    "    .head(5)\n"
                    "    .rename_axis('country')\n"
                    "    .reset_index(name='count')\n"
                    ")"
                )
            if user_query == "average cases":
                return "result = df['confirmed_cases'].mean()"
            if user_query == "trend over time":
                return (
                    "result = (\n"
                    "    df.groupby('date', dropna=False)['confirmed_cases']\n"
                    "    .sum()\n"
                    "    .reset_index()\n"
                    "    .sort_values(by='date')\n"
                    ")"
                )
            raise AssertionError(f"Unexpected query: {user_query}")

        with mock.patch("backend.services.agent_engine.generate_code_with_llm", side_effect=_llm_side_effect):
            first = run_analysis_agent(self.df, "top countries")
            second = run_analysis_agent(self.df, "average cases")
            third = run_analysis_agent(self.df, "trend over time")

        self._print_response("top countries", first)
        self._print_response("average cases", second)
        self._print_response("trend over time", third)

        self.assertEqual(captured_contexts[0], [])
        self.assertEqual(captured_contexts[1][0]["query"], "top countries")
        self.assertEqual(captured_contexts[2][-1]["query"], "average cases")
        self.assertEqual(first["suggestions"], ["Group data by category", "Analyze trend over time"])
        self.assertEqual(second["suggestions"], ["Group data by category", "Analyze trend over time"])
        self.assertEqual(third["suggestions"], ["Group data by category"])
        self.assertEqual(len(agent_engine.analysis_memory.history), 3)
        self.assertLessEqual(len(agent_engine.analysis_memory.get_context()), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
