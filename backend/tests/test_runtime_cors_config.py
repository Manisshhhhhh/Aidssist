import os
import unittest
from unittest import mock

from backend.aidssist_runtime.config import get_settings


class RuntimeCorsConfigTests(unittest.TestCase):
    def tearDown(self):
        get_settings.cache_clear()

    def test_default_cors_origins_include_common_local_ports(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            cors_origins = set(get_settings().cors_origins)

        self.assertIn("http://localhost:5173", cors_origins)
        self.assertIn("http://127.0.0.1:5174", cors_origins)
        self.assertIn("http://localhost:8080", cors_origins)
        self.assertIn("http://127.0.0.1", cors_origins)

    def test_production_defaults_to_wildcard_cors_when_not_configured(self):
        with mock.patch.dict(os.environ, {"AIDSSIST_ENV": "production"}, clear=True):
            get_settings.cache_clear()
            self.assertEqual(get_settings().cors_origins, ("*",))

    def test_port_falls_back_to_platform_port_environment_variable(self):
        with mock.patch.dict(os.environ, {"PORT": "9090"}, clear=True):
            get_settings.cache_clear()
            self.assertEqual(get_settings().api_port, 9090)


if __name__ == "__main__":
    unittest.main()
