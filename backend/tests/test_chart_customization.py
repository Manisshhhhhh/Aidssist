import unittest

import pandas as pd

from backend.chart_customization import ChartCustomization, build_custom_result_chart, get_chart_customization_options


class ChartCustomizationTests(unittest.TestCase):
    def test_build_custom_result_chart_supports_bar_configuration(self):
        table = pd.DataFrame(
            {
                "region": ["East", "West", "East"],
                "sales": [100, 200, 50],
            }
        )

        chart_spec = build_custom_result_chart(
            table,
            ChartCustomization(
                kind="bar",
                x_column="region",
                y_column="sales",
                aggregation="sum",
                palette="blue",
                title="Sales by Region",
            ),
        )

        self.assertIsNotNone(chart_spec)
        self.assertEqual(chart_spec.kind, "bar")
        self.assertEqual(chart_spec.color, "#38BDF8")
        self.assertEqual(chart_spec.title, "Sales by Region")
        self.assertEqual(chart_spec.data.loc[chart_spec.data["region"] == "East", "sales"].iloc[0], 150)

    def test_build_custom_result_chart_supports_histogram_configuration(self):
        table = pd.DataFrame({"sales": [100, 200, 300]})

        chart_spec = build_custom_result_chart(
            table,
            ChartCustomization(
                kind="histogram",
                x_column="sales",
                y_column="sales",
                palette="green",
            ),
        )

        self.assertIsNotNone(chart_spec)
        self.assertEqual(chart_spec.kind, "histogram")
        self.assertEqual(chart_spec.color, "#22C55E")

    def test_get_chart_customization_options_reports_column_groups(self):
        table = pd.DataFrame(
            {
                "order_date": ["2024-01-01", "2024-01-02"],
                "sales": [100, 200],
                "region": ["East", "West"],
            }
        )

        options = get_chart_customization_options(table)

        self.assertIn("sales", options["numeric_columns"])
        self.assertIn("region", options["categorical_columns"])
        self.assertIn("order_date", options["datetime_columns"])


if __name__ == "__main__":
    unittest.main()
