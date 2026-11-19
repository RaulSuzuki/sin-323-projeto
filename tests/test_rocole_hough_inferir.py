"""Testes de rocole_hough_inferir.py: da foto solta (sem mascara) ao diagnostico, com um modelo pequeno treinado em
folhas sinteticas. O SAM e a ResNet-50 de verdade ficam de fora; entram falsos quando o teste precisa deles."""
import unittest

import cv2
import numpy as np

import rocole_hough as rh
import rocole_hough_inferir as hi
import rocole_hough_treinar as ht
from test_rocole_hough import MANCHAS, cena

VINHO = (40, 40, 150)  # BGR: manchas da classe "acaro" das folhas sinteticas


def _manchas_ao_acaso(rng, n):
    return [(int(rng.integers(650, 1270)), int(rng.integers(400, 680))) for _ in range(n)]


def modelo_sintetico():
    """Modelo de dois niveis treinado em 24 folhas sinteticas: sem manchas (saudavel), manchas laranja (ferrugem)
    e manchas vinho menores (acaro)."""
    rng = np.random.default_rng(0)
    dados, ids, classes, grupos = {}, [], [], []
    for i in range(24):
        classe = ht.CLASSES[i % 3]
        img, m = cena(0.5, manchas=_manchas_ao_acaso(rng, 5) if classe == "rust" else [])
        if classe == "red_spider_mite":
            for x, y in _manchas_ao_acaso(rng, 5):
                cv2.circle(img, (x // 2, y // 2), 4, VINHO, -1)
        e = rh.extrair(img, m, 0.9)
        dados[f"f{i}"] = {k: e[k] for k in ("circulos", "at_circulos", "at_folha")}
        ids.append(f"f{i}"); classes.append(classe); grupos.append(i // 6)
    # nivel 2 logistico: com 24 fotos, o gradient boosting (minimo de 20 amostras por folha) nao consegue dividir
    modelo = ht.ajustar(ids, classes, grupos, dados, "completo", "logistica")
    modelo.update({"param1": rh.PARAM1, "param2": rh.PARAM2, "limiar": 0.5, "fusao": None})
    return modelo


class ComModelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modelo = modelo_sintetico()
        cls.diag = hi.Diagnosticador(modelo=cls.modelo, usar_sam=False, usar_resnet=False)


class TestDiagnosticar(ComModelo):
    def test_folha_com_ferrugem(self):
        img, _ = cena(0.5)
        r = self.diag.diagnosticar(img)
        self.assertIs(r["ferrugem"], True)
        self.assertEqual(sorted(r["p"]), sorted(ht.CLASSES))
        self.assertAlmostEqual(sum(r["p"].values()), 1.0, places=5)
        self.assertGreaterEqual(r["p"]["rust"], r["limiar"])
        self.assertEqual(r["mascara"].shape, img.shape[:2])
        self.assertEqual(r["mascara"].dtype, bool)
        self.assertEqual(len(r["circulos"]), len(MANCHAS))
        for x, y in MANCHAS:  # circulos em px da foto recebida (a cena esta pela metade)
            d = min(np.hypot(c[0] - x / 2, c[1] - y / 2) for c in r["circulos"])
            self.assertLess(d, 5)
        self.assertTrue(all(0.0 <= c[3] <= 1.0 for c in r["circulos"]))

    def test_folha_saudavel(self):
        img, _ = cena(0.5, manchas=[])
        r = self.diag.diagnosticar(img)
        self.assertIs(r["ferrugem"], False)
        self.assertEqual(r["circulos"], [])
        self.assertIn("nenhum círculo", r["aviso"])

    def test_sem_sam_o_contorno_vem_da_cor_e_o_aviso_diz(self):
        r = self.diag.diagnosticar(cena(0.5)[0])
        self.assertEqual(r["confianca"], 0.0)
        self.assertIn("cor", r["aviso"])

    def test_foto_em_retrato(self):
        img, _ = cena(0.5, retrato=True)
        r = self.diag.diagnosticar(img)
        self.assertIs(r["ferrugem"], True)
        h, w = img.shape[:2]
        self.assertTrue(all(0 <= c[0] < w and 0 <= c[1] < h for c in r["circulos"]))
        self.assertEqual(len(r["circulos"]), len(MANCHAS))

    def test_foto_grande_e_reduzida_e_os_circulos_voltam_ao_tamanho_dela(self):
        img, _ = cena(2.2)  # 4224 x 2376
        r = self.diag.diagnosticar(img)
        self.assertEqual(r["mascara"].shape, img.shape[:2])
        self.assertEqual(len(r["circulos"]), len(MANCHAS))
        for x, y in MANCHAS:
            c = min(r["circulos"], key=lambda c: np.hypot(c[0] - 2.2 * x, c[1] - 2.2 * y))
            self.assertLess(np.hypot(c[0] - 2.2 * x, c[1] - 2.2 * y), 12)
            self.assertAlmostEqual(c[2], 2.2 * 14, delta=0.4 * 2.2 * 14)

    def test_imagem_sem_folha_nao_quebra_e_avisa(self):
        for cor in (255, 0, 128):
            r = self.diag.diagnosticar(np.full((300, 400, 3), cor, np.uint8))
            self.assertIs(r["ferrugem"], False)
            self.assertEqual(r["circulos"], [])
            self.assertIn("nenhuma folha", r["aviso"])

    def test_imagem_invalida_levanta_value_error(self):
        for ruim in (np.zeros((10, 10, 3), np.uint8), np.zeros((300, 400), np.uint8), None):
            with self.assertRaises(ValueError):
                self.diag.diagnosticar(ruim)


class TestSamEFusao(ComModelo):
    def test_com_sam_usa_a_mascara_e_a_confianca_dele(self):
        img, m = cena(0.5)
        chamadas = []

        def sam_falso(img_rgb):
            chamadas.append(img_rgb.shape)
            return np.stack([m, m, m]), np.array([0.61, 0.5, 0.4])

        r = hi.Diagnosticador(modelo=self.modelo, sam=sam_falso, usar_resnet=False).diagnosticar(img)
        self.assertEqual(chamadas, [img.shape])
        self.assertAlmostEqual(r["confianca"], 0.61, places=5)
        self.assertNotIn("cor", r["aviso"])
        np.testing.assert_array_equal(r["mascara"], m)

    def test_sam_sem_folha_no_centro_cai_no_contorno_pela_cor(self):
        img, m = cena(0.5)
        vazio = np.zeros((3,) + m.shape, bool)
        r = hi.Diagnosticador(modelo=self.modelo, sam=lambda _: (vazio, np.array([0.9, 0.9, 0.9])),
                              usar_resnet=False).diagnosticar(img)
        self.assertEqual(r["confianca"], 0.0)
        self.assertIn("cor", r["aviso"])
        self.assertIs(r["ferrugem"], True)

    def test_fusao_e_a_media_com_a_resnet_e_usa_o_limiar_da_fusao(self):
        img, _ = cena(0.5)
        so_hough = self.diag.diagnosticar(img)
        modelo = {**self.modelo, "fusao": {"w": 0.5, "limiar": 0.99}}
        r = hi.Diagnosticador(modelo=modelo, usar_sam=False,
                              resnet=lambda _: np.array([0.8, 0.1, 0.1])).diagnosticar(img)
        self.assertAlmostEqual(r["p"]["rust"], (so_hough["p"]["rust"] + 0.1) / 2, places=5)
        self.assertAlmostEqual(r["p_resnet"]["healthy"], 0.8, places=5)
        self.assertAlmostEqual(r["p_hough"]["rust"], so_hough["p"]["rust"], places=5)
        self.assertEqual(r["limiar"], 0.99)
        self.assertIs(r["ferrugem"], False)

    def test_fusao_sem_resnet_responde_so_com_o_hough_e_avisa(self):
        modelo = {**self.modelo, "fusao": {"w": 0.5, "limiar": 0.99}}
        r = hi.Diagnosticador(modelo=modelo, usar_sam=False, usar_resnet=False).diagnosticar(cena(0.5)[0])
        self.assertEqual(r["limiar"], 0.5)
        self.assertIsNone(r["p_resnet"])
        self.assertIs(r["ferrugem"], True)
        self.assertIn("ResNet-50", r["aviso"])


class TestIncerto(ComModelo):
    def _diag(self, margem, p_fundida=0.55, com_resnet=True):
        """Fusao com limiar 0,5 e uma ResNet falsa ajustada para a media dar p_fundida de ferrugem."""
        img, _ = cena(0.5)
        p_h = self.diag.diagnosticar(img)["p"]["rust"]
        r = 2 * p_fundida - p_h
        modelo = {**self.modelo, "fusao": {"w": 0.5, "limiar": 0.5}, "recusa": {"taxa": 0.2, "margem": margem}}
        d = hi.Diagnosticador(modelo=modelo, usar_sam=False, usar_resnet=com_resnet,
                              resnet=(lambda _: np.array([1 - r, 0.0, r])) if com_resnet else None)
        return d.diagnosticar(img)

    def test_perto_do_limiar_e_incerto_mas_o_lado_do_limiar_continua_valendo(self):
        r = self._diag(margem=0.2)
        self.assertAlmostEqual(r["p"]["rust"], 0.55, places=5)
        self.assertIs(r["incerto"], True)
        self.assertIs(r["ferrugem"], True)
        self.assertAlmostEqual(r["margem"], 0.2)

    def test_longe_do_limiar_nao_e_incerto(self):
        self.assertIs(self._diag(margem=0.01)["incerto"], False)
        self.assertIs(self._diag(margem=0.2, p_fundida=0.75)["incerto"], False)

    def test_sem_a_resnet_nao_ha_recusa(self):
        r = self._diag(margem=0.9, com_resnet=False)  # a faixa vale para as probabilidades da fusao, nao as do Hough
        self.assertIs(r["incerto"], False)

    def test_modelo_sem_recusa_nunca_e_incerto(self):
        r = self.diag.diagnosticar(cena(0.5)[0])
        self.assertIs(r["incerto"], False)
        self.assertEqual(r["margem"], 0.0)


class TestClassesDoModelo(ComModelo):
    def test_probabilidades_levam_os_nomes_das_classes_do_modelo(self):
        modelo = {**self.modelo, "classes": ["healthy", "other", "rust"]}  # modelo do BRACOL ou dos dois conjuntos
        r = hi.Diagnosticador(modelo=modelo, usar_sam=False, usar_resnet=False).diagnosticar(cena(0.5)[0])
        self.assertEqual(sorted(r["p"]), ["healthy", "other", "rust"])
        self.assertEqual(sorted(r["p_hough"]), ["healthy", "other", "rust"])
        self.assertIs(r["ferrugem"], True)

    def test_modelo_antigo_sem_a_lista_de_classes_usa_as_do_rocole(self):
        modelo = {k: v for k, v in self.modelo.items() if k != "classes"}
        r = hi.Diagnosticador(modelo=modelo, usar_sam=False, usar_resnet=False).diagnosticar(cena(0.5)[0])
        self.assertEqual(sorted(r["p"]), sorted(ht.CLASSES))


class TestAuxiliares(unittest.TestCase):
    def test_reduzir_so_fotos_grandes(self):
        pequena, f = hi.reduzir(np.zeros((1080, 1920, 3), np.uint8))
        self.assertEqual((pequena.shape[:2], f), ((1080, 1920), 1.0))
        grande, f = hi.reduzir(np.zeros((3000, 4000, 3), np.uint8))
        self.assertEqual(max(grande.shape[:2]), hi.LADO_MAX)
        self.assertAlmostEqual(f, hi.LADO_MAX / 4000, places=6)

    def test_entrada_da_resnet_e_sempre_512x288_deitada(self):
        for forma in ((1080, 1920), (1920, 1080), (1000, 1000), (1024, 2048), (3000, 4000)):
            img = np.zeros(forma + (3,), np.uint8)
            self.assertEqual(hi.entrada_resnet(img).size, (512, 288), forma)

    def test_entrada_da_resnet_corta_o_centro(self):
        img = np.zeros((1000, 1000, 3), np.uint8)
        img[400:600] = (0, 0, 255)  # faixa vermelha (BGR) no meio: tem de continuar no meio, em RGB
        saida = np.asarray(hi.entrada_resnet(img))
        self.assertGreater(saida[144, 256, 0], 200)
        self.assertLess(saida[144, 256, 2], 50)

    def test_desenhar_marca_contorno_e_circulos_sem_mudar_o_tamanho(self):
        img, m = cena(0.5)
        r = {"mascara": m, "circulos": [(350.0, 240.0, 7.0, 0.9), (450.0, 300.0, 7.0, 0.1)]}
        saida = hi.desenhar(img, r)
        self.assertEqual(saida.shape, img.shape)
        self.assertFalse(np.array_equal(saida, img))
        self.assertTrue(np.array_equal(img, cena(0.5)[0]))  # a foto recebida nao e alterada
        self.assertEqual(hi.desenhar(img, {"mascara": m, "circulos": []}).shape, img.shape)


if __name__ == "__main__":
    unittest.main()
