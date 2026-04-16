import unittest

import pandas as pd

from backend.time_filter_service import apply_time_filter, detect_time_column, filter_by_time


class TimeFilterTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "order_date": pd.date_range("2025-01-01", periods=420, freq="D"),
                "sales": range(420),
            }
        )

    def test_detect_time_column_prefers_date_like_field(self):
        self.assertEqual(detect_time_column(self.df), "order_date")

    def test_filter_by_time_returns_last_month_window(self):
        filtered = filter_by_time(
            self.df,
            "last_month",
            time_column="order_date",
            reference_time="2025-08-15",
        )

        self.assertEqual(filtered["order_date"].min().date().isoformat(), "2025-07-01")
        self.assertEqual(filtered["order_date"].max().date().isoformat(), "2025-07-31")

    def test_apply_time_filter_supports_custom_range(self):
        filtered = apply_time_filter(
            self.df,
            "custom_range",
            time_column="order_date",
            custom_range={"start_date": "2025-03-10", "end_date": "2025-03-14"},
        )

        self.assertEqual(filtered["order_date"].min().date().isoformat(), "2025-03-10")
        self.assertEqual(filtered["order_date"].max().date().isoformat(), "2025-03-14")
        self.assertEqual(len(filtered), 5)


if __name__ == "__main__":
    unittest.main()
