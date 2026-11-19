"""Testes das partes puras de rocole_hough_avaliar.py: travas do teste e classe predita."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import rocole_hough_avaliar as ha


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

    def test_modelo_de_outra_divisao_para(self):
        meta = self.pasta / "metadados.csv"
        meta.write_text("id,split\nA,train\n", encoding="utf-8")
        ha.conferir_divisao({"metadados_sha256": ha.sha256(meta)}, meta)
        meta.write_text("id,split\nA,test\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            ha.conferir_divisao({"metadados_sha256": "0" * 64}, meta)


class TestClassePredita(unittest.TestCase):
    def test_ferrugem_pelo_limiar_senao_a_maior_das_outras(self):
        p = np.array([[0.5, 0.1, 0.4], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])
        self.assertEqual(ha.classe_predita(p, 0.36).tolist(), ["rust", "red_spider_mite", "rust"])
        self.assertEqual(ha.classe_predita(p, 0.50).tolist(), ["healthy", "red_spider_mite", "rust"])

    def test_ferrugem_com_a_maior_probabilidade_mas_abaixo_do_limiar_nao_e_ferrugem(self):
        p = np.array([[0.30, 0.25, 0.45]])
        self.assertEqual(ha.classe_predita(p, 0.60).tolist(), ["healthy"])


class TestF1PorFolha(unittest.TestCase):
    def test_conta_folhas_e_calcula_f1_de_ferrugem(self):
        import pandas as pd
        # 4 folhas com face (H) e verso (E); a ferrugem "existe" se aparece em algum lado
        df = pd.DataFrame({
            "id": ["C1P1H1", "C1P1E1", "C1P2H1", "C1P2E1", "C1P3H1", "C1P3E1", "C1P4H1", "C1P4E1"],
            "classe": ["rust", "healthy", "healthy", "rust", "healthy", "healthy", "healthy", "healthy"],
            "classe_predita": ["rust", "healthy", "healthy", "healthy", "rust", "healthy", "healthy", "healthy"]})
        t = ha.f1_por_folha(df).set_index("estrategia")
        self.assertEqual(t.loc["so_face", ["folhas", "vp", "fn", "fp"]].tolist(), [4, 1, 1, 1])
        self.assertAlmostEqual(t.loc["so_face", "f1"], 0.5, places=6)
        self.assertEqual(t.loc["so_verso", ["vp", "fn", "fp"]].tolist(), [0, 2, 0])
        self.assertEqual(t.loc["so_verso", "f1"], 0.0)
        self.assertEqual(t.loc["duas_fotos", ["vp", "fn", "fp"]].tolist(), [1, 1, 1])


class TestPontuarEm(unittest.TestCase):
    """O modelo de um conjunto pontuado no teste do mesmo conjunto, com fotos sinteticas."""

    def setUp(self):
        import joblib
        import pandas as pd

        import rocole_hough as rh
        import rocole_hough_dados as hd
        import rocole_hough_treinar as ht
        from test_rocole_hough_treinar import dados_sinteticos
        self.tmp = tempfile.TemporaryDirectory()
        pasta = Path(self.tmp.name)
        self.dados, meta = dados_sinteticos()
        meta = meta.assign(classe=meta["classe"].replace({"red_spider_mite": "other"}), lado="", origem="x",
                           arquivo="", split=["train" if g < 10 else "val" if g < 12 else "test" for g in meta["grupo"]])
        self.csv = pasta / "meta.csv"
        meta.to_csv(self.csv, index=False)
        self.conj = hd.Conjunto("x", meta, ["healthy", "other", "rust"], pasta / "saida", pasta / "x.joblib", [],
                                pasta / "conf.csv", [self.csv])
        tv = meta[meta["split"] != "test"]
        modelo = ht.ajustar(tv["id"], tv["classe"], tv["grupo"], self.dados, "completo", nomes=self.conj.classes)
        modelo.update({"param1": rh.PARAM1, "param2": rh.PARAM2, "limiar": 0.5, "classes": self.conj.classes,
                       "conjunto": "x", "metadados_sha256": self.conj.sha256()})
        joblib.dump(modelo, self.conj.artefato)
        self.original = ht.atributos_do_conjunto
        ht.atributos_do_conjunto = lambda conj, p1=None, p2=None: self.dados
        self.n_teste = int((meta["split"] == "test").sum())

    def tearDown(self):
        import rocole_hough_treinar as ht
        ht.atributos_do_conjunto = self.original
        self.tmp.cleanup()

    def test_grava_metricas_e_predicoes_e_so_pontua_uma_vez(self):
        import pandas as pd
        r = ha.pontuar_em(self.conj)
        self.assertEqual((r["conjunto"], r["n_teste"]), ("x", self.n_teste))
        self.assertEqual(r["classes"], ["healthy", "other", "rust"])
        self.assertGreater(r["ferrugem"]["f1"], 0.8)
        self.assertEqual(sorted(r["taxa_ferrugem_por_classe"]), ["healthy", "other", "rust"])
        self.assertEqual(np.array(r["matriz_confusao"]).shape, (3, 3))
        self.assertEqual(int(np.array(r["matriz_confusao"]).sum()), self.n_teste)
        arq = self.conj.pasta / "resultados_teste.json"
        self.assertTrue(arq.exists())
        pred = pd.read_csv(self.conj.pasta / "predicoes_teste.csv")
        self.assertEqual(len(pred), self.n_teste)
        for col in ("id", "classe", "grupo", "p_rust", "ferrugem"):
            self.assertIn(col, pred.columns)
        self.assertTrue((self.conj.pasta / "teste_intervalos_confianca.csv").exists())
        with self.assertRaises(SystemExit):
            ha.pontuar_em(self.conj)
        ha.pontuar_em(self.conj, refazer=True)

    def test_modelo_treinado_com_outra_divisao_para(self):
        self.csv.write_text(self.csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            ha.pontuar_em(self.conj)

    def test_nao_ha_teste_cruzado_nem_matriz(self):
        with self.assertRaises(SystemExit):
            ha.main(["--dataset", "bracol", "--modelo", "rocole"])
        with self.assertRaises(SystemExit):
            ha.main(["--matriz"])


class TestIndicePredito(unittest.TestCase):
    def test_indice_predito(self):
        p = np.array([[0.5, 0.1, 0.4], [0.2, 0.5, 0.3], [0.30, 0.25, 0.45]])
        self.assertEqual(ha.indice_predito(p, 0.36).tolist(), [2, 1, 2])
        self.assertEqual(ha.indice_predito(p, 0.60).tolist(), [0, 1, 0])


class TestBracol(unittest.TestCase):
    def test_wilson(self):
        inf, sup = ha.wilson(50, 100)
        self.assertAlmostEqual(inf, 0.4038, places=3)
        self.assertAlmostEqual(sup, 0.5962, places=3)
        inf, sup = ha.wilson(0, 20)
        self.assertEqual(inf, 0.0)
        self.assertAlmostEqual(sup, 0.1611, places=3)
        self.assertTrue(all(np.isnan(v) for v in ha.wilson(0, 0)))

    def test_grupos_separam_so_ferrugem_saudavel_e_outras_doencas(self):
        import pandas as pd
        meta = pd.DataFrame({"id": range(1, 9),
                             "miner": [0, 0, 1, 0, 0, 1, 0, 0], "rust": [1, 0, 1, 0, 1, 0, 0, 0],
                             "phoma": [0, 0, 0, 0, 0, 0, 1, 0], "cercospora": [0, 0, 0, 0, 0, 0, 0, 1]})
        g = ha.grupos_bracol(meta, n_outras=2)
        self.assertEqual(g["so_ferrugem"]["id"].tolist(), [1, 5])
        self.assertEqual(g["saudavel"]["id"].tolist(), [2, 4])
        self.assertEqual(len(g["outras_doencas_sem_ferrugem"]), 2)  # sorteadas entre 6, 7 e 8
        self.assertTrue(set(g["outras_doencas_sem_ferrugem"]["id"]) <= {6, 7, 8})
        self.assertNotIn(3, pd.concat(g.values())["id"].tolist())  # ferrugem + bicho-mineiro nao entra em nenhum

    def test_modelos_comparados_no_teste(self):
        self.assertEqual([n for n, _ in ha.modelos_comparados(False)], ["hough", "regras", "resnet50"])
        self.assertEqual([n for n, _ in ha.modelos_comparados(True)], ["fusao", "hough", "regras", "resnet50"])


if __name__ == "__main__":
    unittest.main()
