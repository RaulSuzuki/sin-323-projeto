"""Testes de hough_avaliar.py: passo 1, travas do teste, metricas binarias e o teste do conjunto."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import hough_avaliar as ha


class TestPasso1(unittest.TestCase):
    def test_doenca_pela_maior_e_incerto_abaixo_do_limite(self):
        y = np.array([True, True, False, False, True])
        p = np.array([0.95, 0.55, 0.10, 0.45, 0.30])
        r = ha.passo_1(y, p, limite=0.6)
        self.assertAlmostEqual(r["acerto"], 4 / 5)  # 0,45 e 0,30 viram saudavel; so a de 0,30 tinha doenca
        self.assertAlmostEqual(r["doenca_achada"], 2 / 3)
        self.assertAlmostEqual(r["saudavel_certa"], 1.0)
        self.assertAlmostEqual(r["incerto"], 2 / 5)  # 0,55 e 0,45
        self.assertEqual((r["vp"], r["fn"], r["fp"], r["vn"]), (2, 1, 0, 2))


class TestTravas(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pasta = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_teste_ja_pontuado_para(self):
        arq = self.pasta / "resultados_teste.json"
        ha.conferir_primeira_vez(arq)  # ainda nao existe: segue
        arq.write_text("{}", encoding="utf-8")
        with self.assertRaises(SystemExit):
            ha.conferir_primeira_vez(arq)
        ha.conferir_primeira_vez(arq, refazer=True)  # de proposito


class TestMetricasBinarias(unittest.TestCase):
    def test_metricas_binarias(self):
        m = ha.metricas_binarias(np.array([1, 1, 0, 0, 0]), np.array([1, 0, 1, 0, 0]))
        self.assertAlmostEqual(m["sensibilidade"], 0.5)
        self.assertAlmostEqual(m["especificidade"], 2 / 3)
        self.assertAlmostEqual(m["acuracia_balanceada"], (0.5 + 2 / 3) / 2)
        self.assertAlmostEqual(m["acuracia"], 0.6)
        self.assertAlmostEqual(m["precisao"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.5)

    def test_bootstrap_binario_reamostra_grupos(self):
        y = np.ones(20, int)
        pred = np.r_[np.ones(10, int), np.zeros(10, int)]
        grupos = np.r_[np.zeros(10, int), np.ones(10, int)]
        tab = ha.bootstrap_binario(y, pred, grupos)
        self.assertAlmostEqual(tab.loc["sensibilidade", "estimativa"], 0.5)
        self.assertEqual((tab.loc["sensibilidade", "IC95_inf"], tab.loc["sensibilidade", "IC95_sup"]), (0.0, 1.0))


class TestPontuarEm(unittest.TestCase):
    """O modelo do conjunto pontuado no teste do mesmo conjunto, com fotos sinteticas."""

    def setUp(self):
        import joblib

        import hough as rh
        import hough_dados as hd
        import hough_treinar as ht
        from test_hough_treinar import dados_sinteticos
        self.tmp = tempfile.TemporaryDirectory()
        pasta = Path(self.tmp.name)
        self.dados, meta = dados_sinteticos()
        meta = meta.assign(classe=meta["classe"].replace({"other": "disease"}), origem="x", arquivo="",
                           split=["train" if g < 10 else "val" if g < 12 else "test" for g in meta["grupo"]])
        self.csv = pasta / "meta.csv"
        meta.to_csv(self.csv, index=False)
        self.conj = hd.Conjunto("x", meta, ["healthy", "disease"], pasta / "saida", pasta / "x.joblib", [],
                                pasta / "conf.csv", [self.csv])
        tv = meta[meta["split"] != "test"]
        modelo = ht.ajustar(tv["id"], tv["classe"], tv["grupo"], self.dados, self.conj.classes, "disease")
        modelo.update({"param1": rh.PARAM1, "param2": rh.PARAM2, "limiar": 0.5, "conjunto": "x",
                       "metadados_sha256": self.conj.sha256()})
        joblib.dump(modelo, self.conj.artefato)
        self.original = ht.atributos_do_conjunto
        ht.atributos_do_conjunto = lambda conj: self.dados
        self.n_teste = int((meta["split"] == "test").sum())

    def tearDown(self):
        import hough_treinar as ht
        ht.atributos_do_conjunto = self.original
        self.tmp.cleanup()

    def test_grava_metricas_e_predicoes_e_so_pontua_uma_vez(self):
        import pandas as pd
        r = ha.pontuar_em(self.conj)
        self.assertEqual((r["conjunto"], r["n_teste"]), ("x", self.n_teste))
        self.assertEqual(r["classes"], ["healthy", "disease"])
        t = r["teste"]
        self.assertGreater(t["acerto"], 0.8)
        self.assertEqual(t["vp"] + t["fn"] + t["fp"] + t["vn"], self.n_teste)
        self.assertEqual(sorted(r["ic95"]), ["acerto", "doenca_achada", "saudavel_certa"])
        self.assertNotIn("ferrugem", json.dumps(r))
        arq = self.conj.pasta / "resultados_teste.json"
        self.assertTrue(arq.exists())
        pred = pd.read_csv(self.conj.pasta / "predicoes_teste.csv")
        self.assertEqual(len(pred), self.n_teste)
        for col in ("id", "classe", "grupo", "p_healthy", "p_disease", "doenca", "incerto"):
            self.assertIn(col, pred.columns)
        self.assertTrue((self.conj.pasta / "teste_intervalos_confianca.csv").exists())
        with self.assertRaises(SystemExit):
            ha.pontuar_em(self.conj)
        ha.pontuar_em(self.conj, refazer=True)

    def test_modelo_treinado_com_outra_divisao_para(self):
        self.csv.write_text(self.csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            ha.pontuar_em(self.conj)

    def test_opcoes_desconhecidas_param(self):
        with self.assertRaises(SystemExit):
            ha.main(["--dataset", "bracol", "--modelo", "outro"])
        with self.assertRaises(SystemExit):
            ha.main(["--matriz"])


if __name__ == "__main__":
    unittest.main()
