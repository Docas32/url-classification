import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline_producao
from main import treinar_modelo


class PipelineProducaoTests(unittest.TestCase):
    def test_main_reads_back_rows_from_the_table_it_writes(self) -> None:
        modelo, vetorizador = treinar_modelo()
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "analytics.db")
            output = io.StringIO()
            with (
                patch.object(pipeline_producao, "DB_NAME", db_path),
                patch.object(
                    pipeline_producao,
                    "carregar_artefatos_ml",
                    return_value=(modelo, vetorizador),
                ),
                contextlib.redirect_stdout(output),
            ):
                pipeline_producao.main()

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT log_id, url_bruta, categoria_predita "
                    "FROM url_classification_results"
                ).fetchall()

        self.assertEqual(len(rows), 5)
        self.assertIn("=== DADOS RECUPERADOS", output.getvalue())


if __name__ == "__main__":
    unittest.main()
