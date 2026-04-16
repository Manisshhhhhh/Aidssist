import unittest

from frontend.step_flow import build_step_label, build_step_states, resolve_active_step


class FrontendStepFlowTests(unittest.TestCase):
    def test_only_upload_is_available_without_data(self):
        states = build_step_states(
            has_loaded_source=False,
            has_cleaned_dataset=False,
            can_forecast=False,
            has_successful_forecast=False,
            forecast_skipped=False,
            can_analyze=False,
            has_exportable_output=False,
        )

        self.assertTrue(states["Upload"].accessible)
        self.assertFalse(states["Clean"].accessible)
        self.assertFalse(states["Explore"].accessible)
        self.assertFalse(states["Forecast"].accessible)
        self.assertFalse(states["Analyze"].accessible)
        self.assertFalse(states["Export"].accessible)

    def test_clean_unlocks_after_loading_source(self):
        states = build_step_states(
            has_loaded_source=True,
            has_cleaned_dataset=False,
            can_forecast=False,
            has_successful_forecast=False,
            forecast_skipped=False,
            can_analyze=False,
            has_exportable_output=False,
        )

        self.assertTrue(states["Clean"].accessible)
        self.assertFalse(states["Explore"].accessible)

    def test_explore_unlocks_after_cleaning(self):
        states = build_step_states(
            has_loaded_source=True,
            has_cleaned_dataset=True,
            can_forecast=True,
            has_successful_forecast=False,
            forecast_skipped=False,
            can_analyze=False,
            has_exportable_output=False,
        )

        self.assertTrue(states["Explore"].accessible)
        self.assertTrue(states["Forecast"].accessible)
        self.assertFalse(states["Analyze"].accessible)

    def test_analyze_unlocks_only_after_forecast_or_skip(self):
        states = build_step_states(
            has_loaded_source=True,
            has_cleaned_dataset=True,
            can_forecast=True,
            has_successful_forecast=True,
            forecast_skipped=False,
            can_analyze=True,
            has_exportable_output=False,
        )

        self.assertTrue(states["Analyze"].accessible)
        self.assertFalse(states["Export"].accessible)

    def test_analyze_unlocks_when_forecast_is_skipped(self):
        states = build_step_states(
            has_loaded_source=True,
            has_cleaned_dataset=True,
            can_forecast=True,
            has_successful_forecast=False,
            forecast_skipped=True,
            can_analyze=True,
            has_exportable_output=False,
        )

        self.assertTrue(states["Analyze"].accessible)

    def test_export_unlocks_after_forecast_or_analysis_output(self):
        states = build_step_states(
            has_loaded_source=True,
            has_cleaned_dataset=True,
            can_forecast=True,
            has_successful_forecast=True,
            forecast_skipped=False,
            can_analyze=True,
            has_exportable_output=True,
        )

        self.assertTrue(states["Export"].accessible)

    def test_locked_steps_fall_back_to_latest_accessible_step(self):
        states = build_step_states(
            has_loaded_source=True,
            has_cleaned_dataset=True,
            can_forecast=True,
            has_successful_forecast=False,
            forecast_skipped=False,
            can_analyze=False,
            has_exportable_output=False,
        )

        resolved_step, reason = resolve_active_step("Analyze", states)

        self.assertEqual(resolved_step, "Forecast")
        self.assertEqual(reason, "Resolve or acknowledge validation issues first.")

    def test_label_marks_locked_steps(self):
        states = build_step_states(
            has_loaded_source=False,
            has_cleaned_dataset=False,
            can_forecast=False,
            has_successful_forecast=False,
            forecast_skipped=False,
            can_analyze=False,
            has_exportable_output=False,
        )

        self.assertEqual(build_step_label(states["Upload"]), "Upload")
        self.assertEqual(build_step_label(states["Clean"]), "Clean (Locked)")


if __name__ == "__main__":
    unittest.main()
