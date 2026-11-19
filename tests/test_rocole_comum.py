"""Testes de rocole_comum.py: pasta de dados configuravel por variavel de ambiente."""
import importlib
import os
import unittest
from pathlib import Path
from unittest import mock

import rocole_comum


class TestDados(unittest.TestCase):
    def tearDown(self):
        importlib.reload(rocole_comum)

    def test_padrao_e_data_na_raiz(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROCOLE_DADOS", None)
            m = importlib.reload(rocole_comum)
            self.assertEqual(m.DADOS, m.RAIZ / "data")

    def test_variavel_de_ambiente_troca_a_pasta(self):
        with mock.patch.dict(os.environ, {"ROCOLE_DADOS": str(Path("X:/outro/data"))}):
            m = importlib.reload(rocole_comum)
            self.assertEqual(m.DADOS, Path("X:/outro/data"))
            self.assertEqual(m.ROCOLE_ZIP, Path("X:/outro/data") / "archive.zip")
            self.assertEqual(m.ROCOLE_CACHE_HOUGH, Path("X:/outro/data") / "cache_rocole_hough")


if __name__ == "__main__":
    unittest.main()
