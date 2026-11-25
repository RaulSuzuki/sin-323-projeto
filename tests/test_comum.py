"""Testes de comum.py: pasta de dados configuravel por variavel de ambiente."""
import importlib
import os
import unittest
from pathlib import Path
from unittest import mock

import comum


class TestDados(unittest.TestCase):
    def tearDown(self):
        importlib.reload(comum)

    def test_padrao_e_data_na_raiz(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CAFE_DADOS", None)
            m = importlib.reload(comum)
            self.assertEqual(m.DADOS, m.RAIZ / "data")

    def test_variavel_de_ambiente_troca_a_pasta(self):
        with mock.patch.dict(os.environ, {"CAFE_DADOS": str(Path("X:/outro/data"))}):
            m = importlib.reload(comum)
            self.assertEqual(m.DADOS, Path("X:/outro/data"))
            self.assertEqual(m.CACHE_HOUGH, Path("X:/outro/data") / "cache_hough")
            self.assertEqual(m.BRACOL_CACHE, Path("X:/outro/data") / "cache_512x256")


if __name__ == "__main__":
    unittest.main()
