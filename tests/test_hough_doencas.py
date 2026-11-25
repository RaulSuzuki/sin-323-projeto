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
from test_hough_inferir import logistica, modelo_sintetico
from test_hough_treinar import dados_sinteticos

NOMES = ["healthy", "disease"]


def folhas_bracol(n_fotos=90):
    """Folhas sinteticas com as colunas do BRACOL. As 'other' das folhas sinteticas recebem bicho-mineiro, phoma ou
    cercosporiose em rodizio (cada uma desloca um atributo da folha); as 'disease' recebem ferrugem, e um terco delas
    tem tambem bicho-mineiro. A classe do modelo de Hough e a do BRACOL: saudavel ou doenca."""
    from hough_dados import classe_bracol, estrato_bracol
    dados, meta = dados_sinteticos(n_fotos)
    meta = meta.assign(split=["train" if g < 10 else "val" if g < 12 else "test" for g in meta["grupo"]])
    for c in hd.DOENCAS:
        meta[c] = 0
    outras = ["miner", "phoma", "cercospora"]
    for i, r in meta.iterrows():
        if r["classe"] == "other":
            d = outras[(i // 3) % 3]
            meta.loc[i, d] = 1
            dados[r["id"]]["at_folha"][outras.index(d)] += 5
        elif r["classe"] == "disease":
            meta.loc[i, "rust"] = 1
            if (i // 3) % 3 == 0:
                meta.loc[i, "miner"] = 1
                dados[r["id"]]["at_folha"][0] += 5
    return dados, meta.assign(classe=classe_bracol(meta), estrato=estrato_bracol(meta))


def hough_falso(meta, dados):
    tv = meta[meta["split"].isin(["train", "val"])].set_index("id")
    n1, _ = ht.ajustar_nivel1(tv.index, tv["classe"], tv["grupo"], dados, NOMES)
    return {"param1": rh.PARAM1, "param2": rh.PARAM2, "nivel1": n1, "classes": NOMES, "positiva": "disease"}


def convnext_falsa(meta, ruido=0.3, seed=0):
    """Probabilidades 'fora da dobra' de uma rede falsa: o rotulo com ruido (id, p_<doenca>)."""
    rng = np.random.default_rng(seed)
    y = meta[hd.DOENCAS].to_numpy(float)
    p = np.clip(y * (1 - ruido) + rng.uniform(0, ruido, y.shape), 0, 1)
    return pd.DataFrame({"id": meta["id"].astype(str).to_numpy(), **{f"p_{d}": p[:, j] for j, d in enumerate(hd.DOENCAS)}})


class ConjFalso:
    def __init__(self, meta, pasta):
        self.meta, self.pasta, self.nome, self.classes, self.positiva = meta, Path(pasta), "bracol", list(NOMES), "disease"

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
        modelos = hd.ajustar_doencas(x, y, logistica)
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

    def test_media_com_a_rede(self):
        p_arv, p_cnn = np.array([[0.8, 0.2, 0.0, 0.4]]), np.array([[0.0, 1.0, 0.0, 0.4]])
        np.testing.assert_allclose(hd.media(p_arv, p_cnn), 0.25 * p_arv + 0.75 * p_cnn)
        np.testing.assert_allclose(hd.media(p_arv, p_cnn, 1.0), p_arv)
        self.assertEqual(hd.PESO_ARVORES, 0.25)

    def test_combinar_usa_a_media_so_com_a_rede(self):
        modelo = {"limiares": [0.5] * 4, "fusao": {"w": 0.25, "limiares": [0.4] * 4}}
        p, lim, usou = hd.combinar(modelo, np.array([0.8, 0.2, 0.0, 0.4]), np.array([0.0, 1.0, 0.0, 0.4]))
        np.testing.assert_allclose(p, [0.2, 0.8, 0.0, 0.4])
        np.testing.assert_allclose(lim, [0.4] * 4)
        self.assertTrue(usou)
        p, lim, usou = hd.combinar(modelo, np.array([0.8, 0.2, 0.0, 0.4]), None)  # sem a rede: so as arvores
        np.testing.assert_allclose(p, [0.8, 0.2, 0.0, 0.4])
        np.testing.assert_allclose(lim, [0.5] * 4)
        self.assertFalse(usou)

    def test_f1_por_dobra_usa_os_limiares_das_outras_dobras(self):
        y = np.array([[1, 0, 1, 0]] * 4 + [[0, 1, 0, 1]] * 4)
        p = y * 0.9 + 0.05
        f1 = hd.f1_macro_por_dobra(y, p, np.array([0, 0, 1, 1, 0, 0, 1, 1]))
        self.assertEqual(len(f1), 2)
        self.assertTrue(all(0 <= v <= 1 for v in f1))


class TestComFolhasSinteticas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dados, cls.meta = folhas_bracol()
        cls.hough = hough_falso(cls.meta, cls.dados)
        cls.tv = cls.meta[cls.meta["split"].isin(["train", "val"])]

    def setUp(self):
        self.pasta = Path(tempfile.mkdtemp())
        self.conj = ConjFalso(self.meta, self.pasta)

    def tearDown(self):
        shutil.rmtree(self.pasta, ignore_errors=True)

    def test_validacao_cruzada_so_com_folhas_com_doenca_do_treino_e_validacao(self):
        oof = hd.oof_doencas(self.meta, self.dados, NOMES, "disease", logistica)
        esperado = hd.com_doenca(self.tv)
        self.assertEqual(sorted(oof["id"]), sorted(esperado["id"]))
        self.assertEqual(sorted(oof["dobra"].unique()), [0, 1, 2])
        y = oof[[f"y_{d}" for d in hd.DOENCAS]].to_numpy(int)
        p = oof[[f"p_arvores_{d}" for d in hd.DOENCAS]].to_numpy()
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        _, r = hd.metricas(y, p, hd.escolher_limiares(y, p))
        # cada doenca desloca um atributo: da para separar. Chutar "tem" em todas as doencas daria F1 macro ~0,43
        self.assertGreater(r["f1_macro"], 0.6)

    def test_cv_final_e_teste_uma_vez(self):
        """cv -> final -> teste: os limiares vem da validacao cruzada, o modelo guarda o nivel 1 por doenca e a media
        com a rede, e o teste e pontuado uma vez so."""
        artefato = self.pasta / "bracol_doencas.joblib"
        dec = hd.rodar_cv(self.conj, self.hough, self.dados, cnn=convnext_falsa(self.tv, ruido=0.2), fabrica=logistica)
        self.assertEqual(dec["peso_arvores"], hd.PESO_ARVORES)
        self.assertEqual(set(dec["limiares"]), set(hd.DOENCAS))
        self.assertEqual(set(dec["limiares_arvores"]), set(hd.DOENCAS))
        self.assertEqual(sorted(dec["fora_da_dobra"]), ["arvores", "media"])
        oof = pd.read_csv(self.pasta / "doencas_cv_oof.csv")
        for col in ("p_arvores_rust", "p_convnext_rust", "p_rust"):
            self.assertIn(col, oof.columns)
        np.testing.assert_allclose(oof["p_rust"], hd.media(oof["p_arvores_rust"], oof["p_convnext_rust"]))

        modelo = hd.ajustar_final(self.conj, self.hough, self.dados, artefato=artefato, fabrica=logistica)
        self.assertEqual(modelo["doencas"], hd.DOENCAS)
        self.assertIsNotNone(modelo["nivel1_doencas"])
        self.assertEqual(modelo["fusao"]["w"], hd.PESO_ARVORES)
        self.assertEqual(modelo["fusao"]["limiares"], [dec["limiares"][d] for d in hd.DOENCAS])
        self.assertEqual((modelo["classes_hough"], modelo["positiva_hough"]), (NOMES, "disease"))
        self.assertTrue(artefato.exists())

        te = hd.com_doenca(self.meta[self.meta["split"] == "test"])
        chamadas = []

        def rede(meta):
            chamadas.append(len(meta))
            return convnext_falsa(meta.reset_index(), ruido=0.2).drop(columns="id").to_numpy()

        r = hd.avaliar_teste(self.conj, self.hough, self.dados, artefato=artefato, n_bootstrap=20, convnext=rede)
        self.assertEqual(chamadas, [len(te)])
        self.assertEqual(r["n"], len(te))
        self.assertIn("arvores_sozinhas", r)
        with open(self.pasta / "doencas_teste.json", encoding="utf-8") as f:
            self.assertIn("ic95", json.load(f))
        pred = pd.read_csv(self.pasta / "doencas_predicoes_teste.csv")
        self.assertIn("p_convnext_rust", pred.columns)
        with self.assertRaises(SystemExit):  # o teste e pontuado uma vez so
            hd.avaliar_teste(self.conj, self.hough, self.dados, artefato=artefato, n_bootstrap=20, convnext=rede)

        i = te["id"].iloc[0]
        p3 = ht._pontuar(self.hough["nivel1"], [i], self.dados, len(NOMES))[i]
        p = hd.prever_foto(modelo, self.dados[i], p3)  # o caminho do aplicativo
        self.assertEqual(list(p), hd.DOENCAS)
        self.assertTrue(all(0 <= v <= 1 for v in p.values()))

    def test_cv_sem_as_previsoes_da_rede_para(self):
        with self.assertRaises(SystemExit):  # sem convnext_oof.csv na pasta
            hd.rodar_cv(self.conj, self.hough, self.dados, fabrica=logistica)
        with self.assertRaises(SystemExit):  # a rede nao cobre todas as folhas
            hd.rodar_cv(self.conj, self.hough, self.dados, cnn=convnext_falsa(self.meta.iloc[:5]), fabrica=logistica)

    def test_final_sem_a_validacao_cruzada_para(self):
        with self.assertRaises(SystemExit):
            hd.ajustar_final(self.conj, self.hough, self.dados)


class TestNivel1PorDoenca(unittest.TestCase):
    """O nivel 1 com as 5 classes e o resumo por doenca."""

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
        dados, meta = folhas_bracol()
        tv = meta[meta["split"].isin(["train", "val"])].set_index("id")
        modelo, p = hd.nivel1_por_doenca(tv, dados)
        self.assertEqual(set(p), set(tv.index))
        for i in tv.index[:5]:
            self.assertEqual(p[i].shape, (len(dados[i]["at_circulos"]), 5))
            np.testing.assert_allclose(p[i].sum(1), 1, rtol=1e-6)
        self.assertEqual(sorted(modelo.classes_), list(range(5)))


class TestNoAplicativo(unittest.TestCase):
    """O Diagnosticador devolve uma probabilidade por doenca quando tem o modelo das doencas."""

    @classmethod
    def setUpClass(cls):
        from sklearn.dummy import DummyClassifier
        cls.modelo = modelo_sintetico()
        n1 = DummyClassifier(strategy="prior").fit(np.zeros((5, len(rh.NOMES_CIRCULO))), [0, 1, 2, 3, 4])
        cls.doencas = {"doencas": hd.DOENCAS, "modelos": [0.1, 0.7, 0.2, 0.4], "limiares": [0.35, 0.7, 0.33, 0.48],
                       "nivel1_doencas": n1, "fusao": {"w": 0.5, "limiares": [0.3] * 4},
                       "classes_hough": cls.modelo["classes"], "positiva_hough": cls.modelo["positiva"],
                       "param1": cls.modelo["param1"], "param2": cls.modelo["param2"]}

    def test_com_a_rede_e_a_media_e_os_limiares_dela(self):
        d = hi.Diagnosticador(modelo=self.modelo, usar_sam=False, doencas=self.doencas,
                              convnext=lambda _: np.array([0.9, 0.1, 0.5, 0.3]))
        r = d.diagnosticar(cena(0.5)[0])
        for doenca, esperado in zip(hd.DOENCAS, [0.5, 0.4, 0.35, 0.35]):
            self.assertAlmostEqual(r["p_doencas"][doenca], esperado)
        self.assertEqual(r["limiares_doencas"]["rust"], 0.3)
        self.assertEqual(r["p_doencas_arvores"]["rust"], 0.7)
        self.assertAlmostEqual(r["p_doencas_convnext"]["miner"], 0.9)
        self.assertNotIn("ConvNeXt", r["aviso"])

    def test_sem_a_rede_responde_com_as_arvores_e_avisa(self):
        r = hi.Diagnosticador(modelo=self.modelo, usar_sam=False, doencas=self.doencas,
                              usar_convnext=False).diagnosticar(cena(0.5)[0])
        self.assertEqual(r["p_doencas"], dict(zip(hd.DOENCAS, [0.1, 0.7, 0.2, 0.4])))
        self.assertEqual(r["limiares_doencas"]["rust"], 0.7)  # os limiares das arvores
        self.assertIsNone(r["p_doencas_convnext"])
        self.assertIn("ConvNeXt-Tiny indisponível", r["aviso"])

    def test_circulos_pontuados_pelo_nivel_1_por_doenca(self):
        r = hi.Diagnosticador(modelo=self.modelo, usar_sam=False, doencas=self.doencas,
                              usar_convnext=False).diagnosticar(cena(0.5)[0])
        self.assertGreater(len(r["circulos"]), 0)
        self.assertEqual(len(r["circulos_doencas"]), len(r["circulos"]))
        for pc in r["circulos_doencas"]:
            self.assertEqual(list(pc), hd.CLASSES5)
            self.assertAlmostEqual(sum(pc.values()), 1.0)

    def test_sem_o_modelo_das_doencas(self):
        for doencas in (None, Path(tempfile.gettempdir()) / "nao_existe_doencas.joblib"):
            r = hi.Diagnosticador(modelo=self.modelo, usar_sam=False, doencas=doencas).diagnosticar(cena(0.5)[0])
            self.assertIsNone(r["p_doencas"])
            self.assertIsNone(r["circulos_doencas"])

    def test_hough_diferente_do_modelo_principal_e_erro(self):
        with self.assertRaises(ValueError):
            hi.Diagnosticador(modelo=self.modelo, usar_sam=False,
                              doencas={**self.doencas, "param2": self.modelo["param2"] + 1})


if __name__ == "__main__":
    unittest.main()
