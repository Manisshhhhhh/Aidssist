import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend import prompt_pipeline


class GeminiConfigurationTests(unittest.TestCase):
    def test_placeholder_key_is_rejected(self):
        self.assertTrue(prompt_pipeline._looks_like_placeholder_api_key("sk-your_real_key_here"))
        self.assertTrue(prompt_pipeline._looks_like_placeholder_api_key("gemini-api-key-here"))
        self.assertTrue(prompt_pipeline._looks_like_placeholder_api_key("gsk_*****masked*****"))

    def test_realistic_key_shape_is_not_flagged_as_placeholder(self):
        key = "AIzaSyValidExampleKey1234567890"
        self.assertFalse(prompt_pipeline._looks_like_placeholder_api_key(key))

    def test_gemini_env_file_value_wins_over_stale_shell_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("GEMINI_API_KEY=from-dotenv-value\n", encoding="utf-8")

            with mock.patch.object(prompt_pipeline, "ENV_FILE_PATH", env_path):
                with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "from-shell-value"}, clear=False):
                    self.assertEqual(prompt_pipeline._get_gemini_api_key(), "from-dotenv-value")
                    self.assertEqual(prompt_pipeline._get_gemini_api_key_source(), ".env")

    def test_shell_value_is_reported_when_env_file_is_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("", encoding="utf-8")

            with mock.patch.object(prompt_pipeline, "ENV_FILE_PATH", env_path):
                with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-api-key-here"}, clear=False):
                    is_ready, message = prompt_pipeline.get_gemini_configuration_status()

        self.assertFalse(is_ready)
        self.assertIn("your shell environment", message)

    def test_provider_configuration_requires_gemini_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("", encoding="utf-8")

            with mock.patch.object(prompt_pipeline, "ENV_FILE_PATH", env_path):
                with mock.patch.dict(os.environ, {}, clear=True):
                    is_ready, message = prompt_pipeline.get_provider_configuration_status()

        self.assertFalse(is_ready)
        self.assertIn("Gemini API key is missing", message)

    def test_provider_configuration_ready_with_gemini_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "GEMINI_API_KEY=AIzaSyValidExampleKey1234567890\n",
                encoding="utf-8",
            )

            with mock.patch.object(prompt_pipeline, "ENV_FILE_PATH", env_path):
                is_ready, message = prompt_pipeline.get_provider_configuration_status()

        self.assertTrue(is_ready)
        self.assertIn("Gemini Flash is ready", message)


if __name__ == "__main__":
    unittest.main()
