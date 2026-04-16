import unittest

from backend.data_quality import CleaningOptions
from frontend.workflow_state import (
    build_cleaning_options_signature,
    is_forecast_result_current,
    is_cleaned_dataset_current,
    should_reuse_cleaning_preview,
)


class FrontendWorkflowStateTests(unittest.TestCase):
    def test_build_cleaning_options_signature_is_stable(self):
        signature_a = build_cleaning_options_signature(
            CleaningOptions(parse_dates=True, fill_text_nulls="missing")
        )
        signature_b = build_cleaning_options_signature(
            {
                "fill_text_nulls": "missing",
                "parse_dates": True,
                "coerce_numeric_text": True,
                "trim_strings": True,
                "drop_duplicates": True,
                "fill_numeric_nulls": "none",
                "drop_null_rows_over": 1.0,
                "drop_null_columns_over": 1.0,
            }
        )

        self.assertEqual(signature_a, signature_b)

    def test_should_reuse_cleaning_preview_requires_matching_source_and_options(self):
        preview_state = {
            "_source_fingerprint": "abc123",
            "_options_signature": "sig-1",
        }

        self.assertTrue(should_reuse_cleaning_preview(preview_state, "abc123", "sig-1"))
        self.assertFalse(should_reuse_cleaning_preview(preview_state, "xyz999", "sig-1"))
        self.assertFalse(should_reuse_cleaning_preview(preview_state, "abc123", "sig-2"))

    def test_is_cleaned_dataset_current_when_source_and_options_match(self):
        self.assertTrue(
            is_cleaned_dataset_current(
                active_dataset={
                    "source_fingerprint": "source-1",
                    "cleaning_options_signature": "sig-1",
                },
                loaded_source_state={"source_fingerprint": "source-1"},
                current_options_signature="sig-1",
                applied_source_fingerprint="source-1",
                applied_options_signature="sig-1",
            )
        )

    def test_is_cleaned_dataset_current_returns_false_when_cleaning_settings_change(self):
        self.assertFalse(
            is_cleaned_dataset_current(
                active_dataset={
                    "source_fingerprint": "source-1",
                    "cleaning_options_signature": "sig-1",
                },
                loaded_source_state={"source_fingerprint": "source-1"},
                current_options_signature="sig-2",
                applied_source_fingerprint="source-1",
                applied_options_signature="sig-1",
            )
        )

    def test_is_forecast_result_current_requires_matching_dataset_and_signature(self):
        self.assertTrue(
            is_forecast_result_current(
                forecast_output={
                    "dataset_key": "sales-key",
                    "source_fingerprint": "source-1",
                    "forecast_config_signature": "forecast-sig-1",
                    "error": None,
                },
                active_dataset={
                    "dataset_key": "sales-key",
                    "source_fingerprint": "source-1",
                },
                current_config_signature="forecast-sig-1",
            )
        )
        self.assertFalse(
            is_forecast_result_current(
                forecast_output={
                    "dataset_key": "sales-key",
                    "source_fingerprint": "source-1",
                    "forecast_config_signature": "forecast-sig-1",
                    "error": None,
                },
                active_dataset={
                    "dataset_key": "sales-key",
                    "source_fingerprint": "source-1",
                },
                current_config_signature="forecast-sig-2",
            )
        )


if __name__ == "__main__":
    unittest.main()
