import os
import tempfile
import unittest

from backend.aidssist_runtime.auth_service import (
    authenticate_user,
    get_user_from_token,
    register_user,
    revoke_token,
)
from backend.aidssist_runtime.config import get_settings
from backend.workflow_store import WorkflowStore


class AuthServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "auth.sqlite3")
        self.original_env = {
            "AIDSSIST_DATABASE_URL": os.getenv("AIDSSIST_DATABASE_URL"),
            "AIDSSIST_SESSION_TTL_HOURS": os.getenv("AIDSSIST_SESSION_TTL_HOURS"),
        }
        os.environ["AIDSSIST_DATABASE_URL"] = f"sqlite:///{self.db_path}"
        os.environ["AIDSSIST_SESSION_TTL_HOURS"] = "24"
        get_settings.cache_clear()

    def tearDown(self):
        get_settings.cache_clear()
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()

    def test_register_login_and_revoke_token(self):
        user, register_token = register_user("founder@aidssist.ai", "supersecure", "Founder")

        self.assertEqual(user.email, "founder@aidssist.ai")
        self.assertEqual(user.display_name, "Founder")
        self.assertIsNotNone(get_user_from_token(register_token))

        logged_in_user, login_token = authenticate_user("founder@aidssist.ai", "supersecure")

        self.assertEqual(logged_in_user.user_id, user.user_id)
        self.assertEqual(get_user_from_token(login_token).user_id, user.user_id)

        self.assertTrue(revoke_token(login_token))
        self.assertIsNone(get_user_from_token(login_token))

    def test_user_scoped_history_links_jobs_and_datasets(self):
        user, _ = register_user("ops@aidssist.ai", "supersecure", "Ops")

        store = WorkflowStore()
        try:
            dataset = store.create_dataset(
                dataset_name="sales.csv",
                dataset_key="sales-key",
                source_fingerprint="source-1",
                source_kind="csv",
                source_label="CSV file: sales.csv",
                object_key="datasets/source-1/sales.csv",
                content_type="text/csv",
                size_bytes=256,
                user_id=user.user_id,
            )
            store.create_job(
                dataset_id=dataset.dataset_id,
                query="top customers by revenue",
                intent="general",
                workflow_context={},
                status="completed",
                analysis_output={"result": {"kind": "text", "value": "ok"}},
                result_summary="Top customers identified.",
                cache_hit=True,
                user_id=user.user_id,
            )

            history = store.list_user_history(user.user_id)
            datasets = store.list_user_datasets(user.user_id)
        finally:
            store.close()

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].dataset_name, "sales.csv")
        self.assertEqual(history[0].result_summary, "Top customers identified.")
        self.assertTrue(history[0].cache_hit)
        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0].dataset_id, dataset.dataset_id)


if __name__ == "__main__":
    unittest.main()
