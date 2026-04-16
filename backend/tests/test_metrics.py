import importlib.util
import unittest
from pathlib import Path


class MetricsRegistrationTests(unittest.TestCase):
    def test_metrics_module_can_be_loaded_twice_without_duplicate_registry_errors(self):
        metrics_path = Path(__file__).resolve().parents[1] / "aidssist_runtime" / "metrics.py"

        first_spec = importlib.util.spec_from_file_location("aidssist_metrics_first", metrics_path)
        first_module = importlib.util.module_from_spec(first_spec)
        first_spec.loader.exec_module(first_module)

        second_spec = importlib.util.spec_from_file_location("aidssist_metrics_second", metrics_path)
        second_module = importlib.util.module_from_spec(second_spec)
        second_spec.loader.exec_module(second_module)

        self.assertIs(first_module.REQUEST_COUNT, second_module.REQUEST_COUNT)
        self.assertIs(first_module.REQUEST_LATENCY, second_module.REQUEST_LATENCY)
        self.assertIs(first_module.QUEUE_DEPTH, second_module.QUEUE_DEPTH)


if __name__ == "__main__":
    unittest.main()
