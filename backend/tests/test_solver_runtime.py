import os
import tempfile
import unittest

from backend.aidssist_runtime.chunking import build_text_chunks
from backend.aidssist_runtime.embedding import deterministic_embedding
from backend.aidssist_runtime.refinement import bounded_refinement_loop
from backend.aidssist_runtime.retrieval import retrieve_workspace_context
from backend.aidssist_runtime.config import get_settings
from backend.workflow_store import WorkflowStore


class SolverRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_env = {
            "AIDSSIST_DATABASE_URL": os.getenv("AIDSSIST_DATABASE_URL"),
            "AIDSSIST_REDIS_URL": os.getenv("AIDSSIST_REDIS_URL"),
            "AIDSSIST_OBJECT_STORE_BACKEND": os.getenv("AIDSSIST_OBJECT_STORE_BACKEND"),
        }
        os.environ["AIDSSIST_DATABASE_URL"] = f"sqlite:///{self.temp_dir.name}/solver-runtime.sqlite3"
        os.environ["AIDSSIST_REDIS_URL"] = ""
        os.environ["AIDSSIST_OBJECT_STORE_BACKEND"] = "local"
        get_settings.cache_clear()

    def tearDown(self):
        get_settings.cache_clear()
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()

    def test_build_text_chunks_respects_chunk_boundaries(self):
        text = "alpha beta gamma delta epsilon " * 180
        chunks = build_text_chunks(file_name="notes.md", text=text, file_kind="document", language="markdown")

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk["content_text"]), get_settings().chunk_size_chars + 50)
            self.assertIn("chunk_index", chunk)

    def test_bounded_refinement_loop_stops_after_first_valid_candidate(self):
        attempt_log: list[int] = []

        def build_candidate(attempt_index: int, feedback: str | None):
            del feedback
            attempt_log.append(attempt_index)
            return {"summary": "ok", "value": attempt_index}

        def validate_candidate(payload):
            passed = payload["value"] >= 1
            return passed, [{"name": "value_check", "status": "passed" if passed else "failed"}], None if passed else "retry"

        candidate, steps, reports = bounded_refinement_loop(
            build_candidate=build_candidate,
            validate_candidate=validate_candidate,
            max_retries=3,
        )

        self.assertEqual(candidate["value"], 1)
        self.assertEqual(attempt_log, [0, 1])
        self.assertEqual(len(steps), 2)
        self.assertEqual(reports[-1]["status"], "passed")

    def test_retrieval_feedback_boosts_ranked_chunk(self):
        with WorkflowStore() as store:
            workspace = store.create_workspace(user_id="user-1", name="Solver")
            asset = store.create_asset(workspace_id=workspace.workspace_id, title="Docs", asset_kind="code")
            weak_chunk = store.record_chunk(
                asset_id=asset.asset_id,
                asset_file_id=None,
                dataset_id=None,
                chunk_index=0,
                title="Parser notes",
                content_text="This parser handles text input and schema parsing.",
                token_count=12,
                metadata={"file_name": "parser.py"},
            )
            strong_chunk = store.record_chunk(
                asset_id=asset.asset_id,
                asset_file_id=None,
                dataset_id=None,
                chunk_index=1,
                title="Sales memory",
                content_text="Sales trends, revenue segments, and monthly forecast memory.",
                token_count=12,
                metadata={"file_name": "sales.md"},
            )
            store.upsert_embedding(
                chunk_id=weak_chunk.chunk_id,
                model_name="test",
                vector=deterministic_embedding("parser schema text"),
            )
            store.upsert_embedding(
                chunk_id=strong_chunk.chunk_id,
                model_name="test",
                vector=deterministic_embedding("sales forecast revenue memory"),
            )
            store.record_feedback_event(
                run_id="run-1",
                chunk_id=strong_chunk.chunk_id,
                event_type="retrieval_rank",
                score=6,
                metadata={"reason": "helpful"},
            )

            trace = retrieve_workspace_context(
                store=store,
                workspace_id=workspace.workspace_id,
                query="sales forecast",
                top_k=2,
            )

        self.assertEqual(trace["items"][0]["chunk_id"], strong_chunk.chunk_id)


if __name__ == "__main__":
    unittest.main()
