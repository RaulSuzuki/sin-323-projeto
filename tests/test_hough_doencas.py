"""Testes de hough_doencas.py (uma probabilidade por doenca do BRACOL) com folhas sinteticas: nada de dados reais nem
de treino de verdade."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import hough as rh
import hough_doencas as hd
import hough_inferir as hi
import hough_treinar as ht
from test_hough import cena
from test_hough_inferir import modelo_sintetico
from test_hough_treinar import dados_sinteticos

NOMES = ["healthy", "other", "rust"]


def folhas_bracol(n_fotos=90):
    """Folhas sinteticas com as colunas do BRACOL. 'other' recebe bicho-mineiro, phoma ou cercosporiose em rodizio
    (cada uma desloca um atributo da folha); um terco das folhas com ferrugem tem tambem bicho-mineiro."""
    dados, meta = dados_sinteticos(n_fotos)
    meta = meta.assign(classe=meta["classe"].replace({"red_spider_mite": "other"}),
                       split=["train" if g < 10 else "val" if g < 12 else "test" for g in meta["grupo"]])
    for c in hd.DOENCAS:
        meta[c] = 0
    outras = ["miner", "phoma", "cercospora"]
    for i, r in meta.iterrows():
        if r["classe"] == "other":
            d = outras[(i // 3) % 3]
            meta.loc[i, d] = 1
            dados[r["id"]]["at_folha"][outras.index(d)] += 5
        elif r["classe"] == "rust":
            meta.loc[i, "rust"] = 1
            if (i // 3) % 3 == 0:
                meta.loc[i, "miner"] = 1
                dados[r["id"]]["at_folha"][0] += 5
    return dados, meta


class ConjFalso:
    def __init__(self, meta, pasta):
        self.meta, self.pasta, self.nome, self.classes = meta, Path(pasta), "bracol", list(NOMES)

    def sha256(self):
        return "divisao-de-teste"


class TestModelo(unittest.TestCase):
    def test_um_classificador_por_doenca_e_probabilidades(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 5))
        y = np.zeros((40, 4), int)
        y[:20, 0] = 1
        y[::2, 1] = 1
        y[:, 3] = 1  # sempre presente: vira uma constante
        modelos = hd.ajustar_doencas(x, y, "logistica")
        self.assertEqual(len(modelos), 4)
        self.assertEqual(modelos[2], 0.0)
        self.assertEqual(modelos[3], 1.0)
        p = hd.prever_doencas(modelos, x)
        self.assertEqual(p.shape, (40, 4))
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertTrue((p[:, 3] == 1.0).all())

    def test_metricas(self):
        y = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
        p = np.array([[0.9, 0.1, 0.0, 0.2], [0.2, 0.8, 0.1, 0.9], [0.8, 0.3, 0.0, 0.1]])
        tab, r = hd.metricas(y, p, [0.5] * 4)
        self.assertEqual(list(tab["doenca"]), hd.DOENCAS)
        self.assertAlmostEqual(r["acerto_exato"], 2 / 3)  # a terceira folha perdeu a ferrugem
        self.assertEqual(r["acerto_mais_provavel"], 1.0)

    def test_regra_de_escolha_do_classificador(self):
        """Substitui a base so com media maior E vitoria em pelo menos 2 das 3 dobras."""
        linhas = []
        for k, (base, cand) in enumerate([(0.9, 0.2), (0.2, 0.9), (0.2, 0.9)]):
            for i in range(6):
                y = [1, 0, 1, 0] if i % 2 else [0, 1, 0, 1]
                bom = np.array(y, float)
                ruim = 1 - bom
                linhas.append({"id": f"{k}{i}", "dobra": k, "grupo": k, **{f"y_{d}": y[j] for j, d in enumerate(hd.DOENCAS)},
                               **{f"p_base_{d}": (bom if base > 0.5 else ruim)[j] for j, d in enumerate(hd.DOENCAS)},
                               **{f"p_cand_{d}": (bom if cand > 0.5 else ruim)[j] for j, d in enumerate(hd.DOENCAS)}})
        oof = pd.DataFrame(linhas)
        tipo, tab = hd.escolher_tipo(oof, ("base", "cand"))
        self.assertEqual(tipo, "cand")
        self.assertEqual(list(tab.columns), ["base", "cand"])


class TestComFolhasSinteticas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dados, cls.meta = folhas_bracol()
        cls.pasta = Path(tempfile.mkdtemp())
        tv = cls.meta[cls.meta["split"].isin(["train", "val"])].set_index("id")
        n1, _ = ht.ajustar_nivel1(tv.index, tv["classe"], tv["grupo"], cls.dados, NOMES)
        cls.hough = {"variante": "completo", "param1": rh.PARAM1, "param2": rh.PARAM2, "nivel1": n1}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.pasta, ignore_errors=True)

    def test_validacao_cruzada_so_com_folhas_com_doenca_do_treino_e_validacao(self):
        oof = hd.oof_doencas(self.meta, self.dados, "completo", NOMES, tipos=("logistica",))
        esperado = hd.com_doenca(self.meta[self.meta["split"].isin(["train", "val"])])
        self.assertEqual(sorted(oof["id"]), sorted(esperado["id"]))
        self.assertEqual(sorted(oof["dobra"].unique()), [0, 1, 2])
        y, p = hd._yp(oof, "logistica")
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        _, r = hd.metricas(y, p, hd.escolher_limiares(y, p))
        self.assertGreater(r["f1_macro"], 0.7)  # cada doenca desloca um atributo: da para separar

    def test_cv_final_e_teste_uma_vez(self):
        conj = ConjFalso(self.meta, self.pasta)
        artefato = self.pasta / "bracol_doencas.joblib"
        dec = hd.rodar_cv(conj, self.hough, self.dados, tipos=("logistica",))
        self.assertEqual(dec["tipo"], "logistica")
        self.assertEqual(set(dec["limiares"]), set(hd.DOENCAS))
        modelo = hd.ajustar_final(conj, self.hough, self.dados, artefato=artefato)
        self.assertEqual(modelo["doencas"], hd.DOENCAS)
        self.assertTrue(artefato.exists())
        r = hd.avaliar_teste(conj, self.hough, self.dados, artefato=artefato, n_bootstrap=30)
        self.assertEqual(r["n"], len(hd.com_doenca(self.meta[self.meta["split"] == "test"])))
        with open(self.pasta / "doencas_teste.json", encoding="utf-8") as f:
            self.assertIn("ic95", json.load(f))
        with self.assertRaises(SystemExit):  # o teste e pontuado uma vez so
            hd.avaliar_teste(conj, self.hough, self.dados, artefato=artefato, n_bootstrap=30)

    def test_final_sem_a_validacao_cruzada_para(self):
        with self.assertRaises(SystemExit):
            hd.ajustar_final(ConjFalso(self.meta, self.pasta / "vazia"), self.hough, self.dados)


class TestNivel1PorDoenca(unittest.TestCase):
    """Experimento do comando comparar: nivel 1 com as 5 classes e o resumo por doenca."""

    @classmethod
    def setUpClass(cls):
        cls.dados, cls.meta = folhas_bracol()
        cls.pasta = Path(tempfile.mkdtemp())
        tv = cls.meta[cls.meta["split"].isin(["train", "val"])].set_index("id")
        cls.tv = tv
        n1, _ = ht.ajustar_nivel1(tv.index, tv["classe"], tv["grupo"], cls.dados, NOMES)
        cls.hough = {"variante": "completo", "param1": rh.PARAM1, "param2": rh.PARAM2, "nivel1": n1}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.pasta, ignore_errors=True)

    def test_rotulo_unico(self):
        meta = pd.DataFrame({"miner": [0, 1, 1, 0], "rust": [0, 0, 1, 1], "phoma": [0, 0, 0, 0], "cercospora": [0, 0, 0, 0]})
        self.assertEqual(list(hd.rotulo_unico(meta)), ["healthy", "miner", None, "rust"])

    def test_resumo_por_doenca(self):
        rng = np.random.default_rng(1)
        at = rng.normal(size=(6, len(rh.NOMES_CIRCULO)))
        p = np.full((6, 5), 0.05)
        p[:, 4] = 0.8  # todos parecem cercosporiose
        p[0, 4], p[0, 1] = 0.1, 0.75  # menos o primeiro, que parece bicho-mineiro
        v = hd.estatisticas_por_doenca(p, at, np.full(6, 10.0))
        self.assertEqual(len(v), len(hd.NOMES_POR_DOENCA))
        nomes = list(hd.NOMES_POR_DOENCA)
        self.assertAlmostEqual(v[nomes.index("cercospora_max")], 0.8, places=5)
        self.assertAlmostEqual(v[nomes.index("cercospora_frac_maior")], 5 / 6, places=5)
        self.assertAlmostEqual(v[nomes.index("miner_max")], 0.75, places=5)
        sem = hd.estatisticas_por_doenca(np.zeros((0, 5)), np.zeros((0, len(rh.NOMES_CIRCULO))), np.zeros(0))
        self.assertTrue((sem == 0).all())

    def test_todas_as_folhas_pontuadas_fora_da_dobra(self):
        modelo, p = hd.nivel1_por_doenca(self.tv, self.dados)
        self.assertEqual(set(p), set(self.tv.index))
        for i in self.tv.index[:5]:
            self.assertEqual(p[i].shape, (len(self.dados[i]["at_circulos"]), 5))
            np.testing.assert_allclose(p[i].sum(1), 1, rtol=1e-6)
        self.assertEqual(sorted(modelo.classes_), list(range(5)))

    def test_comparar_so_na_validacao_e_sem_mexer_no_modelo(self):
        conj = ConjFalso(self.meta, self.pasta)
        hd.rodar_cv(conj, self.hough, self.dados, tipos=("logistica",))
        decisao = (self.pasta / "doencas_decisao.json").read_text(encoding="utf-8")
        res = hd.comparar(conj, self.hough, self.dados, tipos=("logistica",))
        self.assertEqual(list(res["f1_macro_por_dobra"]), ["atual_logistica", "por_doenca_logistica"])
        self.assertIn(res["escolhida"], res["f1_macro_por_dobra"])
        self.assertEqual({r["modelo"] for r in res["por_doenca"]}, {"atual_logistica", "por_doenca_logistica"})
        self.assertEqual((self.pasta / "doencas_decisao.json").read_text(encoding="utf-8"), decisao)
        self.assertFalse((self.pasta / "doencas_teste.json").exists())
        self.assertTrue((self.pasta / "doencas_cv_por_doenca.json").exists())

    def test_cv_comparar_adotar_final_e_teste(self):
        """cv -> comparar -> adotar -> final -> teste: o modelo adotado guarda o nivel 1 por doenca e o teste dele e
        pontuado uma vez so, em doencas_teste.json."""
        pasta = Path(tempfile.mkdtemp())
        try:
            conj = ConjFalso(self.meta, pasta)
            artefato = pasta / "bracol_doencas.joblib"
            hd.rodar_cv(conj, self.hough, self.dados, tipos=("logistica",))
            hd.comparar(conj, self.hough, self.dados, tipos=("logistica",))
            arq = pasta / "doencas_cv_por_doenca.json"
            res = json.loads(arq.read_text(encoding="utf-8"))
            res.update({"substitui_o_atual": True, "escolhida": "por_doenca_logistica"})  # forca a adocao no teste
            arq.write_text(json.dumps(res), encoding="utf-8")
            dec = hd.adotar(conj)
            self.assertTrue(dec["nivel1_por_doenca"])
            self.assertEqual(dec["versao"], "v2")
            self.assertEqual(dec["anterior"]["tipo"], "logistica")
            with self.assertRaises(SystemExit):  # nao adota duas vezes
                hd.adotar(conj)
            modelo = hd.ajustar_final(conj, self.hough, self.dados, artefato=artefato)
            self.assertIsNotNone(modelo["nivel1_doencas"])
            self.assertEqual(modelo["versao"], "v2")
            r = hd.avaliar_teste(conj, self.hough, self.dados, artefato=artefato, n_bootstrap=20)
            self.assertGreater(r["n"], 0)
            with open(pasta / "doencas_teste.json", encoding="utf-8") as f:
                self.assertTrue(json.load(f)["nivel1_por_doenca"])
            self.assertTrue((pasta / "doencas_predicoes_teste.csv").exists())
            self.assertEqual(sorted(p.name for p in pasta.glob("*_v*")), [])  # nenhum arquivo com versao no nome
            with self.assertRaises(SystemExit):  # o teste e pontuado uma vez so
                hd.avaliar_teste(conj, self.hough, self.dados, artefato=artefato, n_bootstrap=20)
            i = self.meta[self.meta["split"] == "test"]["id"].iloc[0]
            p3 = ht._pontuar(self.hough["nivel1"], [i], self.dados)[i]
            p = hd.prever_foto(modelo, self.dados[i], p3)
            self.assertEqual(list(p), hd.DOENCAS)
            self.assertTrue(all(0 <= v <= 1 for v in p.values()))
        finally:
            shutil.rmtree(pasta, ignore_errors=True)

    def test_adotar_sem_vitoria_na_regra_para(self):
        pasta = Path(tempfile.mkdtemp())
        try:
            conj = ConjFalso(self.meta, pasta)
            (pasta / "doencas_decisao.json").write_text(json.dumps({"tipo": "hgb", "variante": "completo"}), encoding="utf-8")
            (pasta / "doencas_cv_por_doenca.json").write_text(json.dumps({"substitui_o_atual": False, "escolhida": "atual_hgb"}),
                                                              encoding="utf-8")
            (pasta / "doencas_cv_por_doenca_oof.csv").write_text("id\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                hd.adotar(conj)
        finally:
            shutil.rmtree(pasta, ignore_errors=True)

    def test_comparar_sem_a_validacao_cruzada_atual_para(self):
        with self.assertRaises(SystemExit):
            hd.comparar(ConjFalso(self.meta, self.pasta / "vazia"), self.hough, self.dados, tipos=("logistica",))


class TestNoAplicativo(unittest.TestCase):
    """O Diagnosticador devolve uma probabilidade por doenca quando tem o modelo das doencas."""

    @classmethod
    def setUpClass(cls):
        cls.modelo = modelo_sintetico()
        cls.doencas = {"doencas": hd.DOENCAS, "modelos": [0.1, 0.7, 0.2, 0.4], "limiares": [0.35, 0.7, 0.33, 0.48],
                       "variante": "completo", "param1": cls.modelo["param1"], "param2": cls.modelo["param2"]}

    def test_com_o_modelo_das_doencas(self):
        d = hi.Diagnosticador(modelo=self.modelo, usar_sam=False, usar_resnet=False, doencas=self.doencas)
        r = d.diagnosticar(cena(0.5)[0])
        self.assertEqual(r["p_doencas"], dict(zip(hd.DOENCAS, [0.1, 0.7, 0.2, 0.4])))
        self.assertEqual(r["limiares_doencas"]["rust"], 0.7)
        self.assertIsNone(r["circulos_doencas"])  # sem nivel 1 por doenca, os circulos ficam com o modelo de Hough

    def test_circulos_pontuados_por_doenca_com_o_nivel_1_por_doenca(self):
        from sklearn.dummy import DummyClassifier
        n1 = DummyClassifier(strategy="prior").fit(np.zeros((5, len(rh.NOMES_CIRCULO))), [0, 1, 2, 3, 4])
        d = hi.Diagnosticador(modelo=self.modelo, usar_sam=False, usar_resnet=False,
                              doencas={**self.doencas, "nivel1_doencas": n1})
        r = d.diagnosticar(cena(0.5)[0])
        self.assertGreater(len(r["circulos"]), 0)
        self.assertEqual(len(r["circulos_doencas"]), len(r["circulos"]))
        for pc in r["circulos_doencas"]:
            self.assertEqual(list(pc), hd.CLASSES5)
            self.assertAlmostEqual(sum(pc.values()), 1.0)

    def test_sem_o_modelo_das_doencas(self):
        for doencas in (None, Path(tempfile.gettempdir()) / "nao_existe_doencas.joblib"):
            r = hi.Diagnosticador(modelo=self.modelo, usar_sam=False, usar_resnet=False,
                                  doencas=doencas).diagnosticar(cena(0.5)[0])
            self.assertIsNone(r["p_doencas"])

    def test_hough_diferente_do_modelo_principal_e_erro(self):
        with self.assertRaises(ValueError):
            hi.Diagnosticador(modelo=self.modelo, usar_sam=False, usar_resnet=False,
                              doencas={**self.doencas, "param2": self.modelo["param2"] + 1})


if __name__ == "__main__":
    unittest.main()
