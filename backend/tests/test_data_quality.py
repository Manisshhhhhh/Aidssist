import io
import unittest

import pandas as pd

from backend.analysis_contract import ensure_analysis_contract_defaults, validate_analysis_request
from backend.cleaning_engine import clean_data
from backend.data_quality import (
    CleaningOptions,
    apply_cleaning_plan,
    build_data_quality_report,
    compute_data_quality,
    generate_data_warnings,
    has_blocking_findings,
    profile_data,
    summarize_findings,
    validate_dataframe,
)


class DataQualityTests(unittest.TestCase):
    def test_validate_dataframe_flags_missingness_duplicates_and_mixed_types(self):
        df = pd.DataFrame(
            {
                "score": ["10", "bad", None, "20"],
                "mostly_missing": [None, None, 1, None],
                "all_missing": [None, None, None, None],
            }
        )
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        findings = validate_dataframe(df)
        warning_count, error_count = summarize_findings(findings)

        self.assertGreaterEqual(warning_count, 2)
        self.assertGreaterEqual(error_count, 1)
        self.assertTrue(has_blocking_findings(findings))
        categories = {finding.category for finding in findings}
        self.assertIn("duplicate_rows", categories)
        self.assertIn("high_missingness", categories)
        self.assertIn("mixed_numeric_text", categories)

    def test_clean_data_handles_messy_csv_missing_values_and_duplicate_heavy_rows(self):
        csv_buffer = io.StringIO(
            "\n".join(
                (
                    "order_date,sales,segment,mostly_missing",
                    "2024-01-01,100,SMB,",
                    "2024-01-02,120,,",
                    "2024-01-02,120,,",
                    "2024-01-03,130,Enterprise,",
                    "2024-01-04,140,Mid-market,",
                    "2024-01-05,9999,Enterprise,1",
                )
            )
        )
        df = pd.read_csv(csv_buffer)

        result = clean_data(df)
        cleaned_df = result["cleaned_df"]

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_df["order_date"]))
        self.assertNotIn("mostly_missing", cleaned_df.columns)
        self.assertEqual(cleaned_df.shape[0], 5)
        self.assertEqual(
            cleaned_df["segment"].tolist(),
            ["SMB", "Unknown", "Enterprise", "Mid-market", "Enterprise"],
        )
        self.assertGreaterEqual(result["outliers"].get("sales", 0), 1)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["quality_score"], 1.0)

    def test_apply_cleaning_plan_produces_structured_cleaning_report(self):
        df = pd.DataFrame(
            {
                "order_date": ["2024-01-01 ", "2024-01-02", "bad-date"],
                "sales": ["100", None, "200"],
                "segment": [" SMB ", None, "Mid-market"],
                "mostly_missing": [None, None, 1],
            }
        )

        result = apply_cleaning_plan(
            df,
            CleaningOptions(
                parse_dates=True,
                coerce_numeric_text=True,
                trim_strings=True,
                drop_duplicates=True,
                fill_numeric_nulls="mean",
                fill_text_nulls="missing",
            ),
        )

        report = result.report or {}

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result.dataframe["order_date"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(result.dataframe["sales"]))
        self.assertEqual(result.dataframe["segment"].tolist(), ["SMB", "Missing", "Mid-market"])
        self.assertNotIn("mostly_missing", result.dataframe.columns)
        self.assertEqual(report.get("missing_handled"), 4)
        self.assertEqual(report.get("duplicates_removed"), 0)
        self.assertEqual(report.get("quality_score"), 1.0)
        self.assertIn("mostly_missing", report.get("columns_dropped", []))
        self.assertIn("order_date", report.get("type_conversions", {}))
        self.assertIn("sales", report.get("type_conversions", {}))
        self.assertIn("segment", report.get("type_conversions", {}))

    def test_profile_data_and_quality_score_capture_missing_mixed_types_and_outliers(self):
        df = pd.DataFrame(
            {
                "order_date": [
                    "2025-01-01",
                    "2025-01-02",
                    "bad-date",
                    None,
                    "2025-01-05",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-08",
                    "2025-01-09",
                    "2025-01-10",
                ],
                "sales": [10, 12, 11, 400, 9, 10, 11, 10, 12, 9],
                "region": ["North", None, "South", "East", None, "North", "South", "East", "North", "South"],
            }
        )

        profile = profile_data(df)
        score = compute_data_quality(profile)
        warnings = generate_data_warnings(profile)

        self.assertIn("region", profile["missing_percent"])
        self.assertIn("order_date", profile["inconsistent_types"])
        self.assertIn("sales", profile["outliers"])
        self.assertEqual(profile["column_classification"]["sales"], "numeric")
        self.assertEqual(profile["column_classification"]["order_date"], "datetime")
        self.assertGreaterEqual(profile["invalid_values"]["order_date"]["count"], 1)
        self.assertIn("sales", profile["summary"]["anomaly_columns"])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)
        self.assertTrue(any("mixed" in warning.lower() for warning in warnings))

    def test_profile_data_skips_boolean_columns_for_outlier_detection(self):
        df = pd.DataFrame(
            {
                "is_active": pd.Series([True, False, True, None, False, True, False, True], dtype="boolean"),
                "sales": [10, 12, 11, 400, 9, 10, 11, 10],
            }
        )

        profile = profile_data(df)

        self.assertIn("is_active", profile["column_types"])
        self.assertNotIn("is_active", profile["outliers"])
        self.assertIn("sales", profile["outliers"])

    def test_build_data_quality_report_returns_structured_output(self):
        df = pd.DataFrame(
            {
                "event_date": [
                    "2025-01-01",
                    "2025-01-02",
                    "bad-date",
                    "2025-01-04",
                    "2025-01-05",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-08",
                ],
                "sales": [10, 12, 11, 400, 9, 10, 11, 10],
                "is_active": pd.Series([True, False, True, None, False, True, False, True], dtype="boolean"),
                "segment": ["SMB", "Enterprise", "SMB", "Mid-market", "SMB", "Enterprise", "SMB", "Mid-market"],
            }
        )

        report = build_data_quality_report(df)

        self.assertIn("data_profile", report)
        self.assertIn("data_quality_score", report)
        self.assertIn("warnings", report)
        self.assertIn("anomalies", report)
        self.assertIn("score", report)
        self.assertLessEqual(report["data_quality_score"], 1.0)
        self.assertEqual(report["data_profile"]["column_classification"]["is_active"], "boolean")
        self.assertNotIn("is_active", report["anomalies"])
        self.assertIn("sales", report["anomalies"])

    def test_validate_analysis_request_exposes_reliability_context_for_prediction(self):
        df = pd.DataFrame(
            {
                "order_date": [f"2025-01-{day:02d}" for day in range(1, 13)],
                "sales": [10, 12, "bad", 400, 9, "bad", 11, 10, 12, 9, "bad", 8],
                "segment": ["SMB", "SMB", None, "Enterprise", "SMB", None, "SMB", "Enterprise", "SMB", None, "SMB", "Enterprise"],
            }
        )

        preflight = validate_analysis_request(
            "predict sales next month",
            df,
            {
                "intent": "prediction",
                "analysis_type": "regression",
                "metric_column": "sales",
                "target_column": "sales",
                "datetime_column": "order_date",
            },
        )

        self.assertIn("data_profile", preflight)
        self.assertIn("data_quality", preflight)
        self.assertIn("data_quality_score", preflight)
        self.assertIn("anomalies", preflight)
        self.assertGreaterEqual(preflight["data_quality_score"], 0.0)
        self.assertLessEqual(preflight["data_quality_score"], 1.0)
        self.assertTrue(
            any(
                "Prediction reliability is reduced" in warning
                or "Prediction inputs include" in warning
                or "Prediction inputs still contain" in warning
                for warning in preflight["warnings"]
            )
        )

    def test_contract_defaults_preserve_cleaning_report(self):
        contract = ensure_analysis_contract_defaults(
            {
                "data_quality": {
                    "score": 8.5,
                    "data_quality_score": 0.85,
                    "warnings": ["Moderate missingness in sales."],
                    "anomalies": {"sales": {"count": 2}},
                    "data_profile": {"column_classification": {"sales": "numeric"}},
                },
                "cleaning_report": {
                    "quality_score": 0.92,
                    "missing_handled": 8,
                    "duplicates_removed": 3,
                    "outliers_detected": 2,
                    "before": {
                        "row_count": 10,
                        "column_count": 4,
                        "missing_cells": 8,
                        "duplicate_rows": 3,
                    },
                    "after": {
                        "row_count": 7,
                        "column_count": 3,
                        "missing_cells": 0,
                        "duplicate_rows": 0,
                    },
                    "outlier_columns": {"sales": 2},
                }
            }
        )

        self.assertEqual(contract["cleaning_report"]["quality_score"], 0.92)
        self.assertEqual(contract["cleaning_report"]["missing_handled"], 8)
        self.assertEqual(contract["cleaning_report"]["duplicates_removed"], 3)
        self.assertEqual(contract["cleaning_report"]["outlier_columns"]["sales"], 2)
        self.assertEqual(contract["cleaning_report"]["after"]["missing_cells"], 0)
        self.assertEqual(contract["data_quality"]["data_quality_score"], 0.85)
        self.assertIn("sales", contract["data_quality"]["anomalies"])
        self.assertEqual(contract["data_quality"]["data_profile"]["column_classification"]["sales"], "numeric")
        self.assertEqual(contract["system_decision"]["selected_mode"], "analysis")


if __name__ == "__main__":
    unittest.main()
