import os
import unittest
from unittest import mock

from backend.aidssist_runtime.config import get_settings


class RuntimeConfigTests(unittest.TestCase):
    def tearDown(self):
        get_settings.cache_clear()

    def test_default_upload_limit_is_500_mb(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            self.assertEqual(get_settings().max_upload_mb, 500)

    def test_upload_limit_can_be_overridden_by_environment(self):
        with mock.patch.dict(os.environ, {"AIDSSIST_MAX_UPLOAD_MB": "750"}, clear=True):
            get_settings.cache_clear()
            self.assertEqual(get_settings().max_upload_mb, 750)


if __name__ == "__main__":
    unittest.main()
