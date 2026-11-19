"""Testes das metricas, do bootstrap por grupo e da escolha de exemplos (rocole_avaliar.py)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rocole_avaliar import (_acc_f1, atualizar_comparacao, bootstrap_por_grupo, epoca_escolhida, escolher_acertos,
                            escolher_erros, grupos_por_id, main, metricas)


class TestMetricas(unittest.TestCase):
    def test_metricas(self):
        m = metricas(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 1]))
        self.assertAlmostEqual(m["acuracia"], 0.75)
        self.assertAlmostEqual(m["f1_rust"], 2 / 3)
        self.assertAlmostEqual(m["f1_red_spider_mite"], 2 / 3)
        self.assertAlmostEqual(m["f1_macro"], (1 + 2 / 3 + 2 / 3) / 3)

    def test_bootstrap_reamostra_grupos(self):
        # grupo 0: 10 acertos; grupo 1: 10 erros. Por grupo, o IC vai de 0 a 1;
        # por imagem seria ~[0.28, 0.72].
        y = np.zeros(20, int)
        pred = np.r_[np.zeros(10, int), np.ones(10, int)]
        grupos = np.r_[np.zeros(10, int), np.ones(10, int)]
        tab = bootstrap_por_grupo(y, pred, grupos)
        self.assertAlmostEqual(tab.loc["acuracia", "estimativa"], 0.5)
        self.assertEqual((tab.loc["acuracia", "IC95_inf"], tab.loc["acuracia", "IC95_sup"]), (0.0, 1.0))
        self.assertTrue(tab.equals(bootstrap_por_grupo(y, pred, grupos)))  # mesma seed, mesmo resultado


class TestExemplos(unittest.TestCase):
    def test_escolher_erros(self):
        df = pd.DataFrame({"id": list("abcde"), "classe_real": ["rust"] * 5,
                           "classe_predita": ["healthy", "healthy", "red_spider_mite", "healthy", "rust"],
                           "p_healthy": [0.9, 0.6, 0.0, 0.7, 0.1], "p_red_spider_mite": [0.05, 0.2, 0.99, 0.1, 0.0],
                           "p_rust": [0.05, 0.2, 0.01, 0.2, 0.9], "acertou": [False] * 4 + [True]})
        self.assertEqual(escolher_erros(df, n=3)["id"].tolist(), ["c", "a", "d"])

    def test_escolher_acertos(self):
        df = pd.DataFrame({"id": [str(i) for i in range(12)],
                           "classe_real": ["healthy"] * 5 + ["rust"] * 5 + ["red_spider_mite"] * 2,
                           "acertou": [True] * 11 + [False]})
        df["classe_predita"] = df["classe_real"]
        sel = escolher_acertos(df, n=3)
        self.assertTrue(sel["acertou"].all())
        self.assertEqual(sel["classe_real"].value_counts().to_dict(),
                         {"healthy": 3, "rust": 3, "red_spider_mite": 1})
        self.assertTrue(sel.equals(escolher_acertos(df, n=3)))


class TestComparacao(unittest.TestCase):
    def test_atualizar_comparacao(self):
        with tempfile.TemporaryDirectory() as d:
            c = Path(d) / "comparacao_modelos.csv"
            atualizar_comparacao(c, {"modelo": "resnet50", "teste_f1_macro": 0.5})
            atualizar_comparacao(c, {"modelo": "resnet50", "teste_f1_macro": 0.8})
            tab = atualizar_comparacao(c, {"modelo": "convnext_tiny", "teste_f1_macro": 0.7})
            self.assertEqual(tab["modelo"].tolist(), ["resnet50", "convnext_tiny"])
            self.assertEqual(pd.read_csv(c).set_index("modelo").loc["resnet50", "teste_f1_macro"], 0.8)


class TestRobustez(unittest.TestCase):
    def test_f1_rapido_igual_ao_sklearn(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            n = int(rng.integers(1, 40))
            y, pred = rng.integers(0, 3, n), rng.integers(0, 3, n)
            m = metricas(y, pred)
            esperado = [m["acuracia"], m["f1_macro"], m["f1_healthy"], m["f1_red_spider_mite"], m["f1_rust"]]
            np.testing.assert_allclose(_acc_f1(y, pred), esperado, atol=1e-12)

    def test_epoca_escolhida(self):
        with tempfile.TemporaryDirectory() as d:
            h = Path(d) / "historico_treino.csv"
            pd.DataFrame({"epoca": [1, 2, 3, 4], "perda_val": [0.9, 0.4, 0.4, 0.5]}).to_csv(h, index=False)
            self.assertEqual(epoca_escolhida(h, pesos_customizados=False), 2)  # primeiro minimo
            self.assertIsNone(epoca_escolhida(h, pesos_customizados=True))  # historico pode ser de outro treino
            self.assertIsNone(epoca_escolhida(Path(d) / "nao_existe.csv", pesos_customizados=False))

    def test_grupos_por_id_nao_depende_da_ordem(self):
        meta = pd.DataFrame({"id": ["a", "b", "c"], "grupo": [10, 20, 30]})
        np.testing.assert_array_equal(grupos_por_id(["c", "a", "b"], meta), [30, 10, 20])
        with self.assertRaises(AssertionError):
            grupos_por_id(["a", "x"], meta)

    def test_main_sem_pesos_da_mensagem_clara(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--pesos", "nao_existe.pt"])
        self.assertIn("rocole_treinar.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
