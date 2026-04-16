import unittest

import pandas as pd

from backend.dashboard_helpers import (
    build_column_insight,
    build_chart_takeaway,
    build_dataset_key,
    profile_analysis_result,
    profile_dataset,
)


class DashboardHelperTests(unittest.TestCase):
    def test_mixed_dataset_profiles_column_types_and_prefers_datetime_chart(self):
        df = pd.DataFrame(
            {
                "order_date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
                "segment": ["SMB", "Enterprise", "SMB", "Mid-market"],
                "sales": [1200, 1800, 950, 2100],
            }
        )

        profile = profile_dataset(df, dataset_name="sales.csv", dataset_key="dataset-1")

        self.assertEqual(profile.dataset_name, "sales.csv")
        self.assertEqual(profile.dataset_key, "dataset-1")
        self.assertEqual(profile.numeric_column_count, 1)
        self.assertEqual(profile.categorical_column_count, 1)
        self.assertEqual(profile.datetime_column_count, 1)
        self.assertEqual(
            profile.column_type_breakdown.set_index("type")["count"].to_dict(),
            {"Numeric": 1, "Categorical": 1, "Datetime": 1},
        )
        self.assertIsNotNone(profile.content_chart)
        self.assertEqual(profile.content_chart.kind, "line")
        self.assertEqual(profile.content_chart.x, "order_date")
        self.assertTrue(profile.overview_charts)
        self.assertTrue(profile.datetime_charts)
        self.assertFalse(profile.data_dictionary.empty)

    def test_no_numeric_columns_falls_back_to_categorical_chart(self):
        df = pd.DataFrame(
            {
                "team": ["North", "South", "North", "East"],
                "status": ["Open", "Closed", "Open", "Open"],
            }
        )

        profile = profile_dataset(df, dataset_name="teams.csv")

        self.assertEqual(profile.numeric_column_count, 0)
        self.assertEqual(profile.categorical_column_count, 2)
        self.assertEqual(profile.datetime_column_count, 0)
        self.assertIsNotNone(profile.content_chart)
        self.assertEqual(profile.content_chart.kind, "bar")
        self.assertEqual(profile.content_chart.x, "team")
        self.assertTrue(profile.categorical_charts)
        self.assertIn("__aidssist_highlight", profile.content_chart.data.columns)

    def test_no_categorical_columns_can_still_fall_back_to_histogram(self):
        df = pd.DataFrame(
            {
                "revenue": [110.5, 98.2, 150.0, 123.7],
                "profit": [12.5, 8.0, 19.1, 13.0],
            }
        )

        profile = profile_dataset(df, dataset_name="finance.csv")

        self.assertEqual(profile.categorical_column_count, 0)
        self.assertEqual(profile.datetime_column_count, 0)
        self.assertEqual(profile.numeric_column_count, 2)
        self.assertIsNotNone(profile.content_chart)
        self.assertEqual(profile.content_chart.kind, "histogram")
        self.assertEqual(profile.content_chart.x, "revenue")
        self.assertTrue(profile.numeric_charts)

    def test_missing_heavy_dataset_tracks_missing_totals_and_columns(self):
        df = pd.DataFrame(
            {
                "region": ["East", None, None, "West"],
                "sales": [100, None, 250, None],
                "owner": [None, None, "Ana", None],
            }
        )

        profile = profile_dataset(df, dataset_name="missing.csv")

        self.assertEqual(profile.missing_cell_count, 7)
        self.assertListEqual(
            profile.missing_by_column["column"].tolist(),
            ["owner", "region", "sales"],
        )
        self.assertListEqual(
            profile.missing_by_column["missing_count"].tolist(),
            [3, 2, 2],
        )

    def test_duplicate_heavy_dataset_counts_duplicate_rows(self):
        df = pd.DataFrame(
            {
                "customer": ["A", "A", "A", "B"],
                "orders": [3, 3, 3, 1],
            }
        )

        profile = profile_dataset(df, dataset_name="duplicates.csv")

        self.assertEqual(profile.duplicate_row_count, 2)

    def test_very_small_single_column_dataset_profiles_without_errors(self):
        df = pd.DataFrame({"score": [42]})

        profile = profile_dataset(df, dataset_name="tiny.csv")

        self.assertEqual(profile.row_count, 1)
        self.assertEqual(profile.column_count, 1)
        self.assertEqual(profile.numeric_column_count, 1)
        self.assertIsNotNone(profile.content_chart)
        self.assertEqual(profile.content_chart.kind, "histogram")

    def test_build_column_insight_tracks_stats_and_chart(self):
        df = pd.DataFrame(
            {
                "region": ["East", "West", "East", None],
                "sales": [100, 200, 150, 175],
            }
        )

        column_insight = build_column_insight(df, "sales")

        self.assertEqual(column_insight.semantic_type, "Numeric")
        self.assertEqual(column_insight.missing_count, 0)
        self.assertEqual(column_insight.unique_count, 4)
        self.assertIsNotNone(column_insight.chart)
        self.assertEqual(column_insight.chart.kind, "histogram")

    def test_profile_analysis_result_prefers_bar_chart_for_category_and_numeric(self):
        result = pd.DataFrame(
            {
                "region": ["East", "West", "North"],
                "sales": [1000, 800, 650],
            }
        )

        result_profile = profile_analysis_result(result)

        self.assertIsNotNone(result_profile.table)
        self.assertIsNotNone(result_profile.chart)
        self.assertEqual(result_profile.chart.kind, "bar")
        self.assertEqual(result_profile.chart.x, "region")
        self.assertEqual(result_profile.chart.y, "sales")

    def test_profile_analysis_result_prefers_line_chart_for_datetime_and_numeric(self):
        result = pd.DataFrame(
            {
                "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "sales": [1000, 1200, 950],
            }
        )

        result_profile = profile_analysis_result(result)

        self.assertIsNotNone(result_profile.chart)
        self.assertEqual(result_profile.chart.kind, "line")
        self.assertEqual(result_profile.chart.x, "order_date")
        self.assertEqual(result_profile.chart.y, "sales")

    def test_chart_takeaway_describes_top_bar_value(self):
        df = pd.DataFrame(
            {
                "team": ["North", "South", "North", "East", "North"],
                "status": ["Open", "Closed", "Open", "Open", "Closed"],
            }
        )

        profile = profile_dataset(df, dataset_name="teams.csv")
        takeaway = build_chart_takeaway(profile.content_chart)

        self.assertIsNotNone(takeaway)
        self.assertIn("North", takeaway)
        self.assertIn("displayed total", takeaway)

    def test_chart_takeaway_describes_line_peak(self):
        result = pd.DataFrame(
            {
                "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "sales": [1000, 1400, 900],
            }
        )

        result_profile = profile_analysis_result(result)
        takeaway = build_chart_takeaway(result_profile.chart)

        self.assertIsNotNone(takeaway)
        self.assertIn("Jan 02, 2024", takeaway)
        self.assertIn("1,400", takeaway)

    def test_chart_takeaway_describes_histogram_distribution(self):
        df = pd.DataFrame({"revenue": [100, 110, 120, 130, 140]})

        profile = profile_dataset(df, dataset_name="finance.csv")
        takeaway = build_chart_takeaway(profile.content_chart)

        self.assertIsNotNone(takeaway)
        self.assertIn("centers around", takeaway)
        self.assertIn("revenue", takeaway.lower())

    def test_profile_analysis_result_returns_metric_for_scalar(self):
        result_profile = profile_analysis_result(42)

        self.assertIsNone(result_profile.chart)
        self.assertEqual(result_profile.metric_value, "42")
        self.assertEqual(result_profile.text_value, "42")

    def test_dataset_key_uses_file_content(self):
        key_one = build_dataset_key("sample.csv", 10, b"alpha")
        key_two = build_dataset_key("sample.csv", 10, b"alpha")
        key_three = build_dataset_key("sample.csv", 10, b"beta")

        self.assertEqual(key_one, key_two)
        self.assertNotEqual(key_one, key_three)


if __name__ == "__main__":
    unittest.main()
