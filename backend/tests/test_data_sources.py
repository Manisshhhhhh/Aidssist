import io
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from backend.data_sources import (
    CSVSourceConfig,
    ExcelSourceConfig,
    SQLSourceConfig,
    build_dataframe_fingerprint,
    deserialize_source_config,
    load_dataframe_from_source,
    persist_file_source_snapshot,
    serialize_source_config,
    validate_sql_source_config,
)


class DataSourceTests(unittest.TestCase):
    def test_csv_source_supports_delimiter_and_encoding(self):
        csv_bytes = "name;value\nAna;10\nBen;20\n".encode("utf-8")
        source = CSVSourceConfig(
            file_name="sample.csv",
            file_bytes=csv_bytes,
            delimiter=";",
            encoding="utf-8",
        )

        loaded = load_dataframe_from_source(source)

        self.assertEqual(loaded.dataset_name, "sample.csv")
        self.assertEqual(loaded.dataframe["name"].tolist(), ["Ana", "Ben"])
        self.assertEqual(loaded.dataframe["value"].tolist(), [10, 20])
        self.assertEqual(loaded.dataset_key, f"sample.csv:{loaded.source_fingerprint}")

    def test_excel_source_supports_named_sheet(self):
        dataframe = pd.DataFrame({"region": ["North", "South"], "sales": [100, 150]})
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            dataframe.to_excel(writer, index=False, sheet_name="Metrics")

        source = ExcelSourceConfig(
            file_name="metrics.xlsx",
            file_bytes=buffer.getvalue(),
            sheet_name="Metrics",
        )

        loaded = load_dataframe_from_source(source)

        self.assertEqual(loaded.dataset_name, "metrics.xlsx")
        self.assertListEqual(loaded.dataframe.columns.tolist(), ["region", "sales"])
        self.assertEqual(loaded.dataframe.shape, (2, 2))

    def test_persisted_file_source_round_trips_from_snapshot(self):
        csv_bytes = b"name,value\nAna,10\n"
        source = CSVSourceConfig(file_name="roundtrip.csv", file_bytes=csv_bytes)

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_source = persist_file_source_snapshot(source, temp_dir)
            self.assertIsNone(saved_source.file_bytes)
            self.assertTrue(os.path.exists(saved_source.snapshot_path))

            serialized = serialize_source_config(saved_source)
            restored = deserialize_source_config(serialized)
            loaded = load_dataframe_from_source(restored)

        self.assertEqual(loaded.dataframe.iloc[0]["name"], "Ana")

    def test_sql_source_validation_rejects_non_select_queries(self):
        source = SQLSourceConfig(
            kind="postgres",
            host="localhost",
            port=5432,
            database="analytics",
            username="demo",
            password="secret",
            query="DELETE FROM orders",
        )

        issues = validate_sql_source_config(source)

        self.assertTrue(any("SELECT" in issue for issue in issues))

    def test_sql_source_validation_accepts_table_mode_for_mysql(self):
        source = SQLSourceConfig(
            kind="mysql",
            host="localhost",
            port=3306,
            database="analytics",
            username="demo",
            password="secret",
            table_name="orders",
            limit=500,
        )

        issues = validate_sql_source_config(source)

        self.assertEqual(issues, [])

    def test_dataframe_fingerprint_changes_with_data(self):
        df_one = pd.DataFrame({"value": [1, 2]})
        df_two = pd.DataFrame({"value": [1, 3]})

        self.assertNotEqual(build_dataframe_fingerprint(df_one), build_dataframe_fingerprint(df_two))

    @unittest.skipUnless(os.getenv("POSTGRES_TEST_URL"), "POSTGRES_TEST_URL not configured")
    def test_live_postgres_integration_placeholder(self):
        self.assertTrue(os.getenv("POSTGRES_TEST_URL"))

    @unittest.skipUnless(os.getenv("MYSQL_TEST_URL"), "MYSQL_TEST_URL not configured")
    def test_live_mysql_integration_placeholder(self):
        self.assertTrue(os.getenv("MYSQL_TEST_URL"))


if __name__ == "__main__":
    unittest.main()
