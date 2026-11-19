"""Testes de rocole_hough_treinar.py: extracao em lote, modelo de dois niveis, limiar, regra de decisao e fusao."""
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import rocole_hough as rh
import rocole_hough_treinar as ht
from test_rocole_hough import MANCHAS, cena


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
        self.zip = self.pasta / "fotos.zip"
        linhas = []
        with zipfile.ZipFile(self.zip, "w") as z:
            for i, manchas in enumerate([MANCHAS, MANCHAS[:2], []]):
                img, m = cena(0.5, manchas=manchas)
                arq = f"coffee___rust/C1P{i}H1.jpg"
                z.writestr(arq, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
                linhas.append({"id": f"C1P{i}H1", "arquivo": arq})
                if i < 2:  # a terceira foto fica sem mascara: contorno pela cor
                    cv2.imwrite(str(self.pasta / f"C1P{i}H1.png"), m.astype(np.uint8) * 255)
        self.meta = pd.DataFrame(linhas)

    def tearDown(self):
        self.tmp.cleanup()

    def test_carregar_mascara_ausente_devolve_none(self):
        self.assertIsNone(ht.carregar_mascara("nao_existe", (540, 960), pastas=[self.pasta]))

    def test_carregar_mascara_amplia_para_a_foto(self):
        m = ht.carregar_mascara("C1P0H1", (1080, 1920), pastas=[self.pasta])
        self.assertEqual(m.shape, (1080, 1920))
        self.assertEqual(m.dtype, bool)
        self.assertAlmostEqual(m.mean(), cena(0.5)[1].mean(), delta=0.005)

    def test_extrair_tudo_devolve_circulos_e_atributos_por_id(self):
        dados = ht.extrair_tudo(self.meta, self.zip, pastas=[self.pasta], confiancas={"C1P0H1": 0.9, "C1P1H1": 0.6},
                                processos=1)
        self.assertEqual(sorted(dados), ["C1P0H1", "C1P1H1", "C1P2H1"])
        for id_, n in [("C1P0H1", len(MANCHAS)), ("C1P1H1", 2), ("C1P2H1", 0)]:
            d = dados[id_]
            self.assertEqual(d["circulos"].shape, (n, 3))
            self.assertEqual(d["at_circulos"].shape, (n, len(rh.NOMES_CIRCULO)))
            self.assertEqual(d["at_folha"].shape, (len(rh.NOMES_FOLHA),))
        conf = lambda i: float(dados[i]["at_folha"][rh.NOMES_FOLHA.index("confianca")])
        self.assertAlmostEqual(conf("C1P1H1"), 0.6, places=5)
        self.assertEqual(conf("C1P2H1"), 0.0)

    def test_extrair_tudo_grava_e_reaproveita_o_cache(self):
        cache = self.pasta / "atributos.joblib"
        a = ht.extrair_tudo(self.meta, self.zip, pastas=[self.pasta], confiancas={}, processos=1, cache=cache)
        self.assertTrue(cache.exists())
        b = ht.extrair_tudo(self.meta, self.pasta / "zip_que_nao_existe.zip", pastas=[self.pasta], confiancas={},
                            processos=1, cache=cache)
        np.testing.assert_array_equal(a["C1P0H1"]["at_circulos"], b["C1P0H1"]["at_circulos"])


class TestEstatisticasSaco(unittest.TestCase):
    def test_saco_de_tres_circulos(self):
        p = np.array([[0.1, 0.1, 0.8], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6]])
        s = dict(zip(ht.NOMES_SACO, ht.estatisticas_saco(p, np.array([10.0, 20.0, 10.0]))))
        self.assertEqual(s["n_circulos"], 3)
        self.assertAlmostEqual(s["max_rust"], 0.8, places=5)
        self.assertAlmostEqual(s["top3_rust"], 1.6 / 3, places=5)
        self.assertAlmostEqual(s["top5_rust"], 1.6 / 3, places=5)  # so ha 3 circulos
        self.assertAlmostEqual(s["media_rust"], 1.6 / 3, places=5)
        self.assertAlmostEqual(s["soma_rust"], 1.6, places=5)
        self.assertAlmostEqual(s["max_healthy"], 0.6, places=5)
        self.assertEqual(s["n_rust_05"], 2)
        self.assertEqual(s["n_rust_07"], 1)
        self.assertAlmostEqual(s["area_rust"], np.pi * (0.8 * 100 + 0.2 * 400 + 0.6 * 100) / rh.LADO_FOLHA ** 2, places=6)

    def test_saco_vazio_e_zero(self):
        s = ht.estatisticas_saco(np.zeros((0, 3)), np.zeros(0))
        self.assertEqual(s.shape, (len(ht.NOMES_SACO),))
        self.assertFalse(s.any())


class TestEstatisticasRicas(unittest.TestCase):
    def test_medias_dos_circulos_mais_parecidos_com_ferrugem(self):
        f = len(rh.NOMES_CIRCULO)
        at = np.zeros((7, f), np.float32)
        at[:, rh.NOMES_CIRCULO.index("raio")] = [10, 20, 30, 40, 50, 60, 70]
        p = np.zeros((7, 3))
        p[:, 2] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.05, 0.05]  # os 5 primeiros sao os mais parecidos com ferrugem
        s = dict(zip(ht.NOMES_RICOS, ht.estatisticas_ricas(p, at)))
        self.assertAlmostEqual(s["top5_raio"], 30.0, places=4)
        self.assertAlmostEqual(s["todos_raio"], 40.0, places=4)
        self.assertAlmostEqual(s["hist_rust_0"], 2 / 7, places=5)  # P(ferrugem) < 0,2
        self.assertAlmostEqual(s["hist_rust_4"], 2 / 7, places=5)  # P(ferrugem) >= 0,8 (0,8 e 0,9)
        self.assertAlmostEqual(sum(s[f"hist_rust_{i}"] for i in range(5)), 1.0, places=5)

    def test_sem_circulos_e_zero(self):
        s = ht.estatisticas_ricas(np.zeros((0, 3)), np.zeros((0, len(rh.NOMES_CIRCULO)), np.float32))
        self.assertEqual(s.shape, (len(ht.NOMES_RICOS),))
        self.assertFalse(s.any())

    def test_variante_rica_junta_tudo(self):
        dados, meta = dados_sinteticos(12)
        ids = meta["id"].tolist()
        p = {c: np.full((6, 3), 1 / 3) for c in ids}
        x = ht.montar_x(ids, dados, p, "rico")
        self.assertEqual(x.shape, (12, len(ht.NOMES_SACO) + len(ht.NOMES_RICOS) + len(rh.NOMES_FOLHA)))


class TestNivel2Media(unittest.TestCase):
    def test_a_classe_mora_num_modulo_que_nunca_roda_como_script(self):
        """Gravado por `python rocole_hough_treinar.py final`, o modelo guardava a classe como __main__.MediaDeModelos
        e nao carregava em nenhum outro programa (teste, aplicativo): 'Can't get attribute MediaDeModelos'."""
        import pickle
        m = ht._nivel2("media")
        self.assertEqual(type(m).__module__, "rocole_hough")
        self.assertEqual(type(pickle.loads(pickle.dumps(m))).__name__, "MediaDeModelos")

    def test_media_de_dois_classificadores_e_probabilidade(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(150, 5))
        y = np.arange(150) % 3
        x[:, 0] += 3 * y
        m = ht._nivel2("media").fit(x, y)
        p = m.predict_proba(x)
        self.assertEqual(list(m.classes_), [0, 1, 2])
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6)
        self.assertGreater((p.argmax(1) == y).mean(), 0.9)
        esperado = (m.modelos[0].predict_proba(x) + m.modelos[1].predict_proba(x)) / 2
        np.testing.assert_allclose(p, esperado, atol=1e-9)


class TestFusao(unittest.TestCase):
    def test_fundir_e_a_media_ponderada(self):
        p_b, p_cnn = np.array([[0.2, 0.2, 0.6], [1.0, 0.0, 0.0]]), np.array([[0.6, 0.2, 0.2], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(ht.fundir(p_b, p_cnn, 0.75), [[0.3, 0.2, 0.5], [0.75, 0.0, 0.25]])
        np.testing.assert_allclose(ht.fundir(p_b, p_cnn, 1.0), p_b)
        np.testing.assert_allclose(ht.fundir(p_b, p_cnn, 0.0), p_cnn)
        np.testing.assert_allclose(ht.fundir(p_b, p_cnn, 0.5).sum(1), 1.0)

    def test_cada_dobra_e_prevista_so_pelo_modelo_que_nao_a_viu(self):
        from rocole_comum import OUT_ROCOLE
        meta = pd.read_csv(OUT_ROCOLE / "metadados.csv")
        d = ht.dobras(meta)
        vistos = {}

        def falso(k, ids):
            vistos[k] = set(ids)
            return {i: np.array([0.1 * k, 0.2, 0.7 - 0.1 * k]) for i in ids}

        r = ht.probs_resnet_oof(meta, prever_dobra=falso)
        self.assertEqual(sorted(vistos), [0, 1, 2])
        for k in range(3):
            self.assertEqual(vistos[k], set(d.index[d == k]))
        self.assertEqual(len(r), len(d))
        self.assertEqual(set(r.columns), {"id", "dobra", "p_healthy", "p_red_spider_mite", "p_rust"})
        linha = r.set_index("id").loc[d.index[d == 2][0]]
        self.assertAlmostEqual(linha["p_rust"], 0.5, places=6)
        self.assertEqual(int(linha["dobra"]), 2)

    def _oofs(self):
        ids = [f"f{i}" for i in range(12)]
        classe = (["rust"] * 2 + ["healthy"] * 2) * 3
        dobra = [0] * 4 + [1] * 4 + [2] * 4
        b = pd.DataFrame({"id": ids, "dobra": dobra, "classe": classe, "grupo": dobra,
                          "p_healthy": [0.1, 0.85, 0.9, 0.7] * 3, "p_red_spider_mite": 0.0,
                          "p_rust": [0.9, 0.15, 0.1, 0.3] * 3})  # o Hough perde a 2a ferrugem de cada dobra
        c = pd.DataFrame({"id": ids, "dobra": dobra, "p_healthy": [0.5, 0.1, 0.9, 0.4] * 3, "p_red_spider_mite": 0.0,
                          "p_rust": [0.5, 0.9, 0.1, 0.6] * 3})  # a ResNet acha as duas, mas alarma na 2a saudavel
        return b, c

    def test_tabela_fusao_tem_os_extremos_e_os_pesos(self):
        b, c = self._oofs()
        tab, oofs = ht.tabela_fusao(b, c, pesos=(0.5,))
        self.assertEqual(tab["variante"].tolist(), ["hough", "fusao_w0.50", "resnet50_com_limiar"])
        np.testing.assert_allclose(oofs["hough"]["p_rust"], b["p_rust"])
        np.testing.assert_allclose(oofs["resnet50_com_limiar"]["p_rust"], c["p_rust"])
        np.testing.assert_allclose(oofs["fusao_w0.50"]["p_rust"], [0.7, 0.525, 0.1, 0.45] * 3)
        f1 = tab.set_index("variante")["f1_limiar_unico"]
        self.assertAlmostEqual(f1["fusao_w0.50"], 1.0, places=6)  # a media separa as duas ferrugens das saudaveis
        self.assertLess(f1["hough"], 1.0)

    def test_tabela_fusao_casa_as_fotos_pelo_id(self):
        b, c = self._oofs()
        tab1, _ = ht.tabela_fusao(b, c, pesos=(0.5,))
        tab2, _ = ht.tabela_fusao(b, c.sample(frac=1, random_state=3), pesos=(0.5,))  # ResNet em outra ordem
        pd.testing.assert_frame_equal(tab1, tab2)

    def test_escolher_fusao_segue_a_regra(self):
        tab = pd.DataFrame({"variante": ["hough", "fusao_w0.25", "fusao_w0.50", "resnet50_com_limiar"],
                            "f1_d0": [0.80, 0.90, 0.85, 0.99], "f1_d1": [0.80, 0.79, 0.85, 0.99],
                            "f1_d2": [0.80, 0.79, 0.79, 0.99], "limiar": [0.3, 0.4, 0.5, 0.6]})
        # w0.25: media maior, mas 1 vitoria; w0.50: media maior e 2 vitorias; a ResNet sozinha nao concorre
        self.assertEqual(ht.escolher_fusao(tab), {"adotada": True, "variante": "fusao_w0.50", "w": 0.5, "limiar": 0.5})
        self.assertEqual(ht.escolher_fusao(tab[tab["variante"] != "fusao_w0.50"])["adotada"], False)


class TestLimiarEDecisao(unittest.TestCase):
    def test_limiar_de_maior_f1_e_no_empate_o_mais_perto_de_meio(self):
        y, p = np.array([0, 0, 1, 1], bool), np.array([0.10, 0.35, 0.40, 0.90])
        self.assertAlmostEqual(ht.escolher_limiar(y, p), 0.40, places=6)  # 0,36 a 0,40 separam; fica 0,40

    def test_limiar_sem_positivos_fica_em_meio(self):
        self.assertAlmostEqual(ht.escolher_limiar(np.zeros(4, bool), np.array([0.1, 0.2, 0.3, 0.4])), 0.5, places=6)

    def test_f1_por_dobra_usa_limiar_das_outras_dobras(self):
        oof = pd.DataFrame({"dobra": [0, 0, 0, 0, 1, 1, 1, 1],
                            "classe": ["rust", "rust", "healthy", "healthy"] * 2,
                            "p_rust": [0.9, 0.8, 0.3, 0.1, 0.45, 0.42, 0.40, 0.1]})
        f1 = ht.f1_ferrugem_por_dobra(oof)
        # dobra 0 pontuada com o limiar da dobra 1 (0,42) e vice-versa (0,5: a dobra 1 nao acha nenhuma ferrugem)
        self.assertAlmostEqual(f1[0], 1.0, places=6)
        self.assertAlmostEqual(f1[1], 0.0, places=6)
        self.assertAlmostEqual(ht.f1_ferrugem_por_dobra(oof, limiar=0.41)[1], 1.0, places=6)

    def test_decidir_exige_media_maior_e_duas_vitorias(self):
        tab = pd.DataFrame({"base": [0.80, 0.80, 0.80], "ganha_2": [0.82, 0.81, 0.79], "ganha_1": [0.90, 0.79, 0.79],
                            "pior": [0.70, 0.81, 0.81]})
        self.assertTrue(ht.decidir(tab, "base", "ganha_2"))
        self.assertFalse(ht.decidir(tab, "base", "ganha_1"))  # media maior, mas so 1 vitoria
        self.assertFalse(ht.decidir(tab, "base", "pior"))  # 2 vitorias, mas media menor


class TestDobras(unittest.TestCase):
    def test_mesmas_dobras_de_rocole_cv(self):
        from rocole_comum import OUT_ROCOLE
        from rocole_cv import dobras_cv
        meta = pd.read_csv(OUT_ROCOLE / "metadados.csv")
        d = ht.dobras(meta)
        self.assertEqual(len(d), int(meta["split"].isin(["train", "val"]).sum()))
        for k, meta_k in enumerate(dobras_cv(meta)):
            self.assertEqual(set(d.index[d == k]), set(meta_k.loc[meta_k["split"] == "test", "id"]))


class TestCvRepetida(unittest.TestCase):
    def test_outra_semente_da_outras_dobras_sem_dividir_plantas(self):
        from rocole_comum import OUT_ROCOLE
        meta = pd.read_csv(OUT_ROCOLE / "metadados.csv")
        a, b = ht.dobras(meta, seed=42), ht.dobras(meta, seed=43)
        self.assertFalse((a == b).all())
        grupo = meta.set_index("id")["grupo"]
        for d in (a, b):
            self.assertEqual(sorted(d.unique()), [0, 1, 2])
            self.assertTrue((pd.DataFrame({"d": d, "g": grupo.loc[d.index]}).groupby("g")["d"].nunique() == 1).all())

    def test_resumo_compara_cada_variante_com_a_base_por_repeticao(self):
        linhas = []
        for variante, por_rep in {"completo": [[0.80] * 3, [0.82] * 3], "x": [[0.81, 0.80, 0.82], [0.81] * 3]}.items():
            for r, f1s in enumerate(por_rep):
                linhas += [{"variante": variante, "repeticao": r, "dobra": k, "f1": v} for k, v in enumerate(f1s)]
        t = ht.resumo_repetida(pd.DataFrame(linhas), base="completo").set_index("variante")
        self.assertAlmostEqual(t.loc["completo", "media"], 0.81, places=6)
        self.assertAlmostEqual(t.loc["completo", "dp_repeticoes"], np.std([0.80, 0.82], ddof=1), places=6)
        self.assertAlmostEqual(t.loc["x", "media"], 0.81, places=6)
        self.assertAlmostEqual(t.loc["x", "dif_base"], 0.0, places=6)  # +0,01 numa repeticao, -0,01 na outra
        self.assertEqual(int(t.loc["x", "repeticoes_ganhas"]), 1)
        self.assertEqual(int(t.loc["x", "repeticoes"]), 2)
        self.assertEqual(int(t.loc["completo", "repeticoes_ganhas"]), 0)
        self.assertAlmostEqual(t.loc["x", "dobras_ganhas"], 2 / 6, places=6)  # 0,81>0,80; 0,82>0,80; rep 1: nenhuma


def dados_sinteticos(n_fotos=90, seed=0):
    """Fotos falsas: cada uma com 6 circulos de atributos aleatorios; nas de ferrugem, 3 circulos tem o atributo 0
    deslocado. Duas fotos por grupo (planta)."""
    rng = np.random.default_rng(seed)
    f = len(rh.NOMES_CIRCULO)
    dados, linhas = {}, []
    for i in range(n_fotos):
        classe = ["healthy", "red_spider_mite", "rust"][i % 3]
        at = rng.normal(size=(6, f)).astype(np.float32)
        if classe == "rust":
            at[:3, 0] += 6
        if classe == "red_spider_mite":
            at[:3, 1] += 6
        dados[f"f{i}"] = {"circulos": np.abs(rng.normal(10, 2, (6, 3))).astype(np.float32), "at_circulos": at,
                          "at_folha": rng.normal(size=len(rh.NOMES_FOLHA)).astype(np.float32)}
        linhas.append({"id": f"f{i}", "classe": classe, "grupo": i // 6})
    return dados, pd.DataFrame(linhas)


class TestOutrosConjuntos(unittest.TestCase):
    """O mesmo modelo com as classes de outro conjunto (saudavel, outra lesao, ferrugem) e fotos lidas de arquivos."""

    def _dados_gerais(self):
        dados, meta = dados_sinteticos()
        meta = meta.assign(classe=meta["classe"].replace({"red_spider_mite": "other"}),
                           split=["train" if g < 12 else "val" for g in meta["grupo"]])
        return dados, meta

    def test_ajusta_e_preve_com_nomes_de_classe_de_outro_conjunto(self):
        dados, meta = self._dados_gerais()
        nomes = ["healthy", "other", "rust"]
        modelo = ht.ajustar(meta["id"], meta["classe"], meta["grupo"], dados, "completo", nomes=nomes)
        p = ht.prever(modelo, meta["id"], dados)
        self.assertEqual(p.shape, (len(meta), 3))
        acertos = (p.argmax(1) == meta["classe"].map(nomes.index).to_numpy()).mean()
        self.assertGreater(acertos, 0.85)

    def test_classe_desconhecida_para_com_mensagem(self):
        dados, meta = self._dados_gerais()
        with self.assertRaises(ValueError):
            ht.ajustar(meta["id"], meta["classe"], meta["grupo"], dados, "completo")  # 'other' nao e classe do RoCoLe

    def test_cv_grava_as_colunas_com_os_nomes_do_conjunto(self):
        dados, meta = self._dados_gerais()
        oof = ht.cv_externa(meta, dados, "so_folha", nomes=["healthy", "other", "rust"])
        self.assertEqual([c for c in oof.columns if c.startswith("p_")], ["p_healthy", "p_other", "p_rust"])
        self.assertEqual(len(oof), len(meta))
        r = ht.resumo_oof(oof)
        self.assertTrue(0.0 <= r["f1_macro_3_classes"] <= 1.0)
        self.assertEqual(sorted(k for k in r if k.startswith("f1_d")), ["f1_d0", "f1_d1", "f1_d2"])

    def test_extrair_tudo_le_fotos_de_arquivos_quando_nao_ha_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            linhas = []
            for i, manchas in enumerate([MANCHAS, []]):
                img, m = cena(0.5, manchas=manchas)
                arq = Path(tmp) / f"{i + 1}.jpg"
                cv2.imwrite(str(arq), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(Path(tmp) / f"{i + 1}.png"), m.astype(np.uint8) * 255)
                linhas.append({"id": str(i + 1), "arquivo": str(arq)})
            dados = ht.extrair_tudo(pd.DataFrame(linhas), None, pastas=[tmp], confiancas={"1": 0.9, "2": 0.9}, processos=1)
        self.assertEqual(sorted(dados), ["1", "2"])
        self.assertEqual(len(dados["1"]["circulos"]), len(MANCHAS))
        self.assertEqual(len(dados["2"]["circulos"]), 0)

    def test_atributos_do_conjunto_usa_o_cache_do_proprio_conjunto(self):
        import joblib

        import rocole_hough_dados as hd
        with tempfile.TemporaryDirectory() as tmp:
            conj = hd.carregar_conjunto("bracol")
            cache = Path(tmp) / "bracol.joblib"
            joblib.dump({"1": "atributos da foto 1"}, cache)
            conj.cache = lambda p1, p2: cache
            self.assertEqual(ht.atributos_do_conjunto(conj), {"1": "atributos da foto 1"})


class TestModeloDoisNiveis(unittest.TestCase):
    def test_aprende_e_devolve_probabilidades(self):
        dados, meta = dados_sinteticos()
        tr, te = meta[meta["grupo"] < 11], meta[meta["grupo"] >= 11]
        modelo = ht.ajustar(tr["id"].tolist(), tr["classe"].tolist(), tr["grupo"].tolist(), dados, "completo")
        p = ht.prever(modelo, te["id"].tolist(), dados)
        self.assertEqual(p.shape, (len(te), 3))
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
        acertos = (p.argmax(1) == te["classe"].map(ht.CLASSES.index).to_numpy()).mean()
        self.assertGreater(acertos, 0.85)

    def test_variantes_usam_so_a_sua_parte(self):
        dados, meta = dados_sinteticos()
        ids = meta["id"].tolist()
        p = {c: np.full((6, 3), 1 / 3) for c in ids}
        self.assertEqual(ht.montar_x(ids, dados, p, "completo").shape[1], len(ht.NOMES_SACO) + len(rh.NOMES_FOLHA))
        self.assertEqual(ht.montar_x(ids, dados, p, "so_circulos").shape[1], len(ht.NOMES_SACO))
        self.assertEqual(ht.montar_x(ids, dados, None, "so_folha").shape[1], len(rh.NOMES_FOLHA))

    def test_foto_sem_circulos_tambem_e_prevista(self):
        dados, meta = dados_sinteticos()
        modelo = ht.ajustar(meta["id"].tolist(), meta["classe"].tolist(), meta["grupo"].tolist(), dados, "completo")
        dados["vazia"] = {"circulos": np.zeros((0, 3), np.float32),
                          "at_circulos": np.zeros((0, len(rh.NOMES_CIRCULO)), np.float32),
                          "at_folha": np.zeros(len(rh.NOMES_FOLHA), np.float32)}
        p = ht.prever(modelo, ["vazia"], dados)
        self.assertEqual(p.shape, (1, 3))
        self.assertTrue(np.isfinite(p).all())

    def test_nivel_1_nunca_pontua_a_propria_planta(self):
        """Com um nivel 1 que decora (1 vizinho) e rotulos ao acaso, a pontuacao fora da dobra fica no acaso; se a
        foto, ou a outra foto da mesma planta (copia identica), entrasse no treino, daria 1."""
        from sklearn.neighbors import KNeighborsClassifier
        rng = np.random.default_rng(1)
        f = len(rh.NOMES_CIRCULO)
        dados, ids, y, grupos = {}, [], [], []
        for g in range(60):
            at = rng.normal(size=(4, f)).astype(np.float32)
            classe = int(rng.integers(0, 3))
            for lado in "HE":  # duas fotos identicas por planta
                dados[f"g{g}{lado}"] = {"circulos": np.ones((4, 3), np.float32), "at_circulos": at.copy(),
                                        "at_folha": np.zeros(len(rh.NOMES_FOLHA), np.float32)}
                ids.append(f"g{g}{lado}"); y.append(classe); grupos.append(g)
        p = ht.pontuar_fora_da_dobra(ids, np.array(y), np.array(grupos), dados,
                                     fabrica=lambda: KNeighborsClassifier(n_neighbors=1))
        p_certa = np.mean([p[i][:, c].mean() for i, c in zip(ids, y)])
        self.assertLess(p_certa, 0.6)


if __name__ == "__main__":
    unittest.main()
