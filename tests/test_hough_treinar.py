"""Testes de hough_treinar.py: extracao em lote, modelo de dois niveis, validacao cruzada e modelo final."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import hough as rh
import hough_treinar as ht
from test_hough import MANCHAS, cena

NOMES3 = ["healthy", "other", "disease"]  # as tres classes das folhas sinteticas (dados_sinteticos)


class TestEscolherMascaraHough(unittest.TestCase):
    def _mascaras(self):
        m = np.zeros((3, 100, 200), bool)
        m[0, 30:70, 60:140] = True  # 16% da foto, no centro
        m[1, 45:55, 95:105] = True  # 0,5%: pequena demais
        m[2] = True  # 100%: grande demais
        return m

    def test_aceita_baixa_confianca_e_devolve_a_confianca(self):
        m, conf = rh.escolher_mascara_hough(self._mascaras(), np.array([0.55, 0.99, 0.99]))
        self.assertAlmostEqual(conf, 0.55, places=5)
        self.assertEqual(int(m.sum()), 40 * 80)

    def test_sem_area_valida_devolve_none(self):
        m, conf = rh.escolher_mascara_hough(self._mascaras()[1:], np.array([0.99, 0.99]))
        self.assertIsNone(m)
        self.assertEqual(conf, 0.0)


class TestExtracaoEmLote(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pasta = Path(self.tmp.name)
        linhas = []
        for i, manchas in enumerate([MANCHAS, MANCHAS[:2], []]):
            img, m = cena(0.5, manchas=manchas)
            arq = self.pasta / f"f{i}.jpg"
            cv2.imwrite(str(arq), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            linhas.append({"id": f"f{i}", "arquivo": str(arq)})
            if i < 2:  # a terceira foto fica sem mascara: contorno pela cor
                cv2.imwrite(str(self.pasta / f"f{i}.png"), m.astype(np.uint8) * 255)
        self.meta = pd.DataFrame(linhas)

    def tearDown(self):
        self.tmp.cleanup()

    def test_carregar_mascara_ausente_devolve_none(self):
        self.assertIsNone(ht.carregar_mascara("nao_existe", (540, 960), pastas=[self.pasta]))

    def test_carregar_mascara_amplia_para_a_foto(self):
        m = ht.carregar_mascara("f0", (1080, 1920), pastas=[self.pasta])
        self.assertEqual(m.shape, (1080, 1920))
        self.assertEqual(m.dtype, bool)
        self.assertAlmostEqual(m.mean(), cena(0.5)[1].mean(), delta=0.005)

    def test_extrair_tudo_devolve_circulos_e_atributos_por_id(self):
        dados = ht.extrair_tudo(self.meta, pastas=[self.pasta], confiancas={"f0": 0.9, "f1": 0.6}, processos=1)
        self.assertEqual(sorted(dados), ["f0", "f1", "f2"])
        for id_, n in [("f0", len(MANCHAS)), ("f1", 2), ("f2", 0)]:
            d = dados[id_]
            self.assertEqual(d["circulos"].shape, (n, 3))
            self.assertEqual(d["at_circulos"].shape, (n, len(rh.NOMES_CIRCULO)))
            self.assertEqual(d["at_folha"].shape, (len(rh.NOMES_FOLHA),))
        conf = lambda i: float(dados[i]["at_folha"][rh.NOMES_FOLHA.index("confianca")])
        self.assertAlmostEqual(conf("f1"), 0.6, places=5)
        self.assertEqual(conf("f2"), 0.0)

    def test_extrair_tudo_grava_e_reaproveita_o_cache(self):
        cache = self.pasta / "atributos.joblib"
        a = ht.extrair_tudo(self.meta, pastas=[self.pasta], confiancas={}, processos=1, cache=cache)
        self.assertTrue(cache.exists())
        sem_fotos = self.meta.assign(arquivo=str(self.pasta / "foto_que_nao_existe.jpg"))
        b = ht.extrair_tudo(sem_fotos, pastas=[self.pasta], confiancas={}, processos=1, cache=cache)
        np.testing.assert_array_equal(a["f0"]["at_circulos"], b["f0"]["at_circulos"])


class TestEstatisticasSaco(unittest.TestCase):
    def test_saco_de_tres_circulos(self):
        p = np.array([[0.1, 0.1, 0.8], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6]])
        nomes = ht.nomes_saco(NOMES3, "disease")
        s = dict(zip(nomes, ht.estatisticas_saco(p, np.array([10.0, 20.0, 10.0]), n_classes=3, i_pos=2)))
        self.assertEqual(s["n_circulos"], 3)
        self.assertAlmostEqual(s["max_doenca"], 0.8, places=5)
        self.assertAlmostEqual(s["top3_doenca"], 1.6 / 3, places=5)
        self.assertAlmostEqual(s["top5_doenca"], 1.6 / 3, places=5)  # so ha 3 circulos
        self.assertAlmostEqual(s["media_doenca"], 1.6 / 3, places=5)
        self.assertAlmostEqual(s["soma_doenca"], 1.6, places=5)
        self.assertAlmostEqual(s["max_healthy"], 0.6, places=5)
        self.assertEqual(s["n_doenca_05"], 2)
        self.assertEqual(s["n_doenca_07"], 1)
        self.assertAlmostEqual(s["area_doenca"], np.pi * (0.8 * 100 + 0.2 * 400 + 0.6 * 100) / rh.LADO_FOLHA ** 2,
                               places=6)

    def test_saco_vazio_e_zero(self):
        s = ht.estatisticas_saco(np.zeros((0, 2)), np.zeros(0))
        self.assertEqual(s.shape, (len(ht.NOMES_SACO),))
        self.assertFalse(s.any())

    def test_saco_de_saudavel_x_doenca_acompanha_a_doenca(self):
        nomes = ht.nomes_saco(["healthy", "disease"], "disease")
        self.assertEqual(nomes, ht.NOMES_SACO)  # o padrao e saudavel ou doenca
        self.assertEqual(nomes[-3:], ["n_doenca_05", "n_doenca_07", "area_doenca"])
        p = np.array([[0.1, 0.9], [0.8, 0.2]])
        s = dict(zip(nomes, ht.estatisticas_saco(p, np.array([10.0, 10.0]), n_classes=2, i_pos=1)))
        self.assertAlmostEqual(s["max_doenca"], 0.9, places=5)
        self.assertEqual(s["n_doenca_05"], 1)


class TestLimiar(unittest.TestCase):
    def test_limiar_de_maior_f1_e_no_empate_o_mais_perto_de_meio(self):
        y, p = np.array([0, 0, 1, 1], bool), np.array([0.10, 0.35, 0.40, 0.90])
        self.assertAlmostEqual(ht.escolher_limiar(y, p), 0.40, places=6)  # 0,36 a 0,40 separam; fica 0,40

    def test_limiar_sem_positivos_fica_em_meio(self):
        self.assertAlmostEqual(ht.escolher_limiar(np.zeros(4, bool), np.array([0.1, 0.2, 0.3, 0.4])), 0.5, places=6)

    def test_f1_macro_por_dobra_usa_a_classe_mais_provavel(self):
        oof = pd.DataFrame({"dobra": [0, 0, 0, 1, 1, 1], "classe": ["healthy", "disease", "disease"] * 2,
                            "p_healthy": [0.9, 0.2, 0.6, 0.7, 0.1, 0.3], "p_disease": [0.1, 0.8, 0.4, 0.3, 0.9, 0.7]})
        f1 = ht.f1_macro_por_dobra(oof)
        self.assertAlmostEqual(f1[1], 1.0)
        self.assertAlmostEqual(f1[0], (2 / 3 + 2 / 3) / 2)  # a terceira folha da dobra 0 virou saudavel


def dados_sinteticos(n_fotos=90, seed=0):
    """Fotos falsas, das classes de NOMES3: cada uma com 6 circulos de atributos aleatorios; nas de 'disease', 3
    circulos tem o atributo 0 deslocado; nas de 'other', o atributo 1. Seis fotos por grupo."""
    rng = np.random.default_rng(seed)
    f = len(rh.NOMES_CIRCULO)
    dados, linhas = {}, []
    for i in range(n_fotos):
        classe = NOMES3[i % 3]
        at = rng.normal(size=(6, f)).astype(np.float32)
        if classe == "disease":
            at[:3, 0] += 6
        if classe == "other":
            at[:3, 1] += 6
        dados[f"f{i}"] = {"circulos": np.abs(rng.normal(10, 2, (6, 3))).astype(np.float32), "at_circulos": at,
                          "at_folha": rng.normal(size=len(rh.NOMES_FOLHA)).astype(np.float32)}
        linhas.append({"id": f"f{i}", "classe": classe, "grupo": i // 6})
    return dados, pd.DataFrame(linhas)


class TestSaudavelOuDoenca(unittest.TestCase):
    """O modelo com as classes do BRACOL (saudavel ou doenca), o padrao."""
    NOMES = ["healthy", "disease"]

    def _dados_gerais(self):
        dados, meta = dados_sinteticos()
        meta = meta.assign(classe=meta["classe"].replace({"other": "disease"}),
                           split=["train" if g < 12 else "val" for g in meta["grupo"]])
        return dados, meta

    def test_ajusta_e_preve_saudavel_x_doenca(self):
        dados, meta = self._dados_gerais()
        modelo = ht.ajustar(meta["id"], meta["classe"], meta["grupo"], dados, nomes=self.NOMES, positiva="disease")
        self.assertEqual((modelo["classes"], modelo["positiva"]), (self.NOMES, "disease"))
        p = ht.prever(modelo, meta["id"], dados)
        self.assertEqual(p.shape, (len(meta), 2))
        acertos = (p.argmax(1) == meta["classe"].map(self.NOMES.index).to_numpy()).mean()
        self.assertGreater(acertos, 0.85)

    def test_modelo_sem_as_chaves_de_classe_usa_o_padrao(self):
        dados, meta = self._dados_gerais()
        modelo = ht.ajustar(meta["id"], meta["classe"], meta["grupo"], dados)
        self.assertEqual((modelo["classes"], modelo["positiva"]), (self.NOMES, "disease"))
        sem_chaves = {k: v for k, v in modelo.items() if k not in ("classes", "positiva")}
        np.testing.assert_allclose(ht.prever(sem_chaves, meta["id"], dados), ht.prever(modelo, meta["id"], dados))

    def test_classe_desconhecida_para_com_mensagem(self):
        dados, meta = dados_sinteticos()
        with self.assertRaises(ValueError):
            ht.ajustar(meta["id"], meta["classe"], meta["grupo"], dados)  # 'other' nao e classe do padrao

    def test_cv_grava_as_colunas_com_os_nomes_do_conjunto(self):
        dados, meta = self._dados_gerais()
        oof = ht.cv_externa(meta, dados, nomes=self.NOMES, positiva="disease")
        self.assertEqual([c for c in oof.columns if c.startswith("p_")], ["p_healthy", "p_disease"])
        self.assertEqual(sorted(oof["id"]), sorted(meta["id"]))
        self.assertEqual(sorted(oof["dobra"].unique()), [0, 1, 2])
        f1 = ht.f1_macro_por_dobra(oof)
        self.assertTrue(((f1 >= 0) & (f1 <= 1)).all())

    def test_dobras_usam_o_estrato_quando_existe(self):
        _, meta = self._dados_gerais()
        meta = meta.assign(estrato=np.where(meta["grupo"] % 2 == 0, "a", "b"))
        d = ht.dobras(meta)
        tv = meta.set_index("id").loc[d.index]
        self.assertTrue((pd.crosstab(d.to_numpy(), tv["estrato"].to_numpy()) > 0).all().all())

    def test_atributos_do_conjunto_usa_o_cache_do_proprio_conjunto(self):
        import joblib

        import hough_dados as hd
        with tempfile.TemporaryDirectory() as tmp:
            conj = hd.carregar_conjunto("bracol")
            cache = Path(tmp) / "bracol.joblib"
            joblib.dump({"1": "atributos da foto 1"}, cache)
            conj.cache = lambda p1, p2: cache
            self.assertEqual(ht.atributos_do_conjunto(conj), {"1": "atributos da foto 1"})


class TestComandos(unittest.TestCase):
    """cv e final num conjunto pequeno de folhas sinteticas."""

    def setUp(self):
        import hough_dados as hd
        self.tmp = tempfile.TemporaryDirectory()
        pasta = Path(self.tmp.name)
        self.dados, meta = dados_sinteticos()
        meta = meta.assign(classe=meta["classe"].replace({"other": "disease"}),
                           split=["train" if g < 12 else "val" for g in meta["grupo"]])
        csv = pasta / "meta.csv"
        meta.to_csv(csv, index=False)
        self.conj = hd.Conjunto("x", meta, ["healthy", "disease"], pasta / "saida", pasta / "x.joblib", [],
                                pasta / "conf.csv", [csv])
        self.original = ht.atributos_do_conjunto
        ht.atributos_do_conjunto = lambda conj: self.dados

    def tearDown(self):
        ht.atributos_do_conjunto = self.original
        self.tmp.cleanup()

    def test_cv_grava_as_previsoes_fora_da_dobra(self):
        oof = ht.rodar_cv(self.conj)
        gravado = pd.read_csv(self.conj.pasta / "cv_oof.csv")
        self.assertEqual(len(gravado), len(self.conj.meta))
        self.assertEqual(list(gravado.columns), ["id", "dobra", "classe", "grupo", "p_healthy", "p_disease"])
        np.testing.assert_allclose(gravado["p_disease"], oof["p_disease"])

    def test_final_grava_o_modelo_com_o_que_o_aplicativo_precisa(self):
        import joblib
        ht.ajustar_final(self.conj)
        modelo = joblib.load(self.conj.artefato)
        for chave in ("nivel1", "nivel2", "classes", "positiva", "param1", "param2", "limiar", "metadados_sha256"):
            self.assertIn(chave, modelo)
        self.assertEqual((modelo["classes"], modelo["limiar"]), (["healthy", "disease"], 0.5))
        self.assertEqual(modelo["metadados_sha256"], self.conj.sha256())
        self.assertEqual(ht.prever(modelo, ["f0"], self.dados).shape, (1, 2))


class TestModeloDoisNiveis(unittest.TestCase):
    def test_aprende_e_devolve_probabilidades(self):
        dados, meta = dados_sinteticos()
        tr, te = meta[meta["grupo"] < 11], meta[meta["grupo"] >= 11]
        modelo = ht.ajustar(tr["id"].tolist(), tr["classe"].tolist(), tr["grupo"].tolist(), dados, NOMES3, "disease")
        p = ht.prever(modelo, te["id"].tolist(), dados)
        self.assertEqual(p.shape, (len(te), 3))
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
        acertos = (p.argmax(1) == te["classe"].map(NOMES3.index).to_numpy()).mean()
        self.assertGreater(acertos, 0.85)

    def test_montar_x_junta_o_resumo_dos_circulos_e_a_folha(self):
        dados, meta = dados_sinteticos()
        ids = meta["id"].tolist()
        p = {c: np.full((6, 2), 0.5) for c in ids}
        x = ht.montar_x(ids, dados, p)
        self.assertEqual(x.shape, (len(ids), len(ht.NOMES_SACO) + len(rh.NOMES_FOLHA)))
        np.testing.assert_allclose(x[:, len(ht.NOMES_SACO):], np.stack([dados[i]["at_folha"] for i in ids]))

    def test_foto_sem_circulos_tambem_e_prevista(self):
        dados, meta = dados_sinteticos()
        modelo = ht.ajustar(meta["id"].tolist(), meta["classe"].tolist(), meta["grupo"].tolist(), dados, NOMES3,
                            "disease")
        dados["vazia"] = {"circulos": np.zeros((0, 3), np.float32),
                          "at_circulos": np.zeros((0, len(rh.NOMES_CIRCULO)), np.float32),
                          "at_folha": np.zeros(len(rh.NOMES_FOLHA), np.float32)}
        p = ht.prever(modelo, ["vazia"], dados)
        self.assertEqual(p.shape, (1, 3))
        self.assertTrue(np.isfinite(p).all())

    def test_nivel_1_nunca_pontua_o_proprio_grupo(self):
        """Com um nivel 1 que decora (1 vizinho) e rotulos ao acaso, a pontuacao fora da dobra fica no acaso; se a
        foto, ou a outra foto do mesmo grupo (copia identica), entrasse no treino, daria 1."""
        from sklearn.neighbors import KNeighborsClassifier
        rng = np.random.default_rng(1)
        f = len(rh.NOMES_CIRCULO)
        dados, ids, y, grupos = {}, [], [], []
        for g in range(60):
            at = rng.normal(size=(4, f)).astype(np.float32)
            classe = int(rng.integers(0, 3))
            for copia in "ab":  # duas fotos identicas por grupo
                dados[f"g{g}{copia}"] = {"circulos": np.ones((4, 3), np.float32), "at_circulos": at.copy(),
                                         "at_folha": np.zeros(len(rh.NOMES_FOLHA), np.float32)}
                ids.append(f"g{g}{copia}"); y.append(classe); grupos.append(g)
        p = ht.pontuar_fora_da_dobra(ids, np.array(y), np.array(grupos), dados,
                                     fabrica=lambda: KNeighborsClassifier(n_neighbors=1), n_classes=3)
        p_certa = np.mean([p[i][:, c].mean() for i, c in zip(ids, y)])
        self.assertLess(p_certa, 0.6)


if __name__ == "__main__":
    unittest.main()
