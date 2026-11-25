"""Testes de hough_inferir.py: da foto solta (sem mascara) ao diagnostico, com um modelo pequeno treinado em
folhas sinteticas. O SAM e as redes de verdade ficam de fora; entram falsas quando o teste precisa delas."""
import unittest

import cv2
import numpy as np

import hough as rh
import hough_inferir as hi
import hough_treinar as ht
from test_hough import MANCHAS, cena

VINHO = (40, 40, 150)  # BGR: manchas da classe "other" das folhas sinteticas
NOMES3 = ["healthy", "other", "disease"]


def logistica():
    """Regressao logistica padronizada: o nivel 2 dos modelos sinteticos, que tem poucas folhas para o gradient
    boosting (minimo de 20 amostras por folha da arvore)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def _manchas_ao_acaso(rng, n):
    return [(int(rng.integers(650, 1270)), int(rng.integers(400, 680))) for _ in range(n)]


def modelo_sintetico():
    """Modelo de dois niveis treinado em 24 folhas sinteticas: sem manchas (healthy), manchas laranja (disease, a
    classe positiva) e manchas vinho menores (other)."""
    rng = np.random.default_rng(0)
    dados, ids, classes, grupos = {}, [], [], []
    for i in range(24):
        classe = NOMES3[i % 3]
        img, m = cena(0.5, manchas=_manchas_ao_acaso(rng, 5) if classe == "disease" else [])
        if classe == "other":
            for x, y in _manchas_ao_acaso(rng, 5):
                cv2.circle(img, (x // 2, y // 2), 4, VINHO, -1)
        e = rh.extrair(img, m, 0.9)
        dados[f"f{i}"] = {k: e[k] for k in ("circulos", "at_circulos", "at_folha")}
        ids.append(f"f{i}"); classes.append(classe); grupos.append(i // 6)
    modelo = ht.ajustar(ids, classes, grupos, dados, NOMES3, "disease", fabrica=logistica)  # 24 fotos: nivel 2 logistico
    modelo.update({"param1": rh.PARAM1, "param2": rh.PARAM2, "limiar": 0.5})
    return modelo


class ComModelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modelo = modelo_sintetico()
        cls.diag = hi.Diagnosticador(modelo=cls.modelo, usar_sam=False)


class TestDiagnosticar(ComModelo):
    def test_folha_com_doenca(self):
        img, _ = cena(0.5)
        r = self.diag.diagnosticar(img)
        self.assertIs(r["positivo"], True)
        self.assertEqual(sorted(r["p"]), sorted(NOMES3))
        self.assertAlmostEqual(sum(r["p"].values()), 1.0, places=5)
        self.assertGreaterEqual(r["p"]["disease"], r["limiar"])
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
        self.assertIs(r["positivo"], False)
        self.assertEqual(r["circulos"], [])
        self.assertIn("nenhum círculo", r["aviso"])

    def test_sem_sam_o_contorno_vem_da_cor_e_o_aviso_diz(self):
        r = self.diag.diagnosticar(cena(0.5)[0])
        self.assertEqual(r["confianca"], 0.0)
        self.assertIn("cor", r["aviso"])

    def test_foto_em_retrato(self):
        img, _ = cena(0.5, retrato=True)
        r = self.diag.diagnosticar(img)
        self.assertIs(r["positivo"], True)
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
            self.assertIs(r["positivo"], False)
            self.assertEqual(r["circulos"], [])
            self.assertIn("nenhuma folha", r["aviso"])

    def test_imagem_invalida_levanta_value_error(self):
        for ruim in (np.zeros((10, 10, 3), np.uint8), np.zeros((300, 400), np.uint8), None):
            with self.assertRaises(ValueError):
                self.diag.diagnosticar(ruim)


class TestSam(ComModelo):
    def test_com_sam_usa_a_mascara_e_a_confianca_dele(self):
        img, m = cena(0.5)
        chamadas = []

        def sam_falso(img_rgb):
            chamadas.append(img_rgb.shape)
            return np.stack([m, m, m]), np.array([0.61, 0.5, 0.4])

        r = hi.Diagnosticador(modelo=self.modelo, sam=sam_falso).diagnosticar(img)
        self.assertEqual(chamadas, [img.shape])
        self.assertAlmostEqual(r["confianca"], 0.61, places=5)
        self.assertNotIn("cor", r["aviso"])
        np.testing.assert_array_equal(r["mascara"], m)

    def test_sam_sem_folha_no_centro_cai_no_contorno_pela_cor(self):
        img, m = cena(0.5)
        vazio = np.zeros((3,) + m.shape, bool)
        r = hi.Diagnosticador(modelo=self.modelo, sam=lambda _: (vazio, np.array([0.9, 0.9, 0.9]))).diagnosticar(img)
        self.assertEqual(r["confianca"], 0.0)
        self.assertIn("cor", r["aviso"])
        self.assertIs(r["positivo"], True)


class TestClassesDoModelo(ComModelo):
    def test_probabilidades_levam_os_nomes_das_classes_do_modelo(self):
        modelo = {**self.modelo, "classes": ["healthy", "x", "y"], "positiva": "y"}  # os nomes gravados no modelo
        r = hi.Diagnosticador(modelo=modelo, usar_sam=False).diagnosticar(cena(0.5)[0])
        self.assertEqual(sorted(r["p"]), ["healthy", "x", "y"])
        self.assertEqual(sorted(r["circulos_classes"][0]), ["healthy", "x", "y"])
        self.assertIs(r["positivo"], True)


class TestAuxiliares(unittest.TestCase):
    def test_reduzir_so_fotos_grandes(self):
        pequena, f = hi.reduzir(np.zeros((1080, 1920, 3), np.uint8))
        self.assertEqual((pequena.shape[:2], f), ((1080, 1920), 1.0))
        grande, f = hi.reduzir(np.zeros((3000, 4000, 3), np.uint8))
        self.assertEqual(max(grande.shape[:2]), hi.LADO_MAX)
        self.assertAlmostEqual(f, hi.LADO_MAX / 4000, places=6)

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
