"""Testes do pipeline de Hough do RoCoLe (rocole_hough.py): folha normalizada, mapa de lesao, circulos e atributos."""
import unittest

import cv2
import numpy as np

import rocole_hough as rh

VERDE = (0, 160, 0)  # BGR
LARANJA = (0, 128, 255)
MARROM = (50, 80, 120)  # fundo (solo)
MANCHAS = [(700, 480), (900, 600), (1100, 450), (1250, 560), (960, 380)]  # centros na cena 1920x1080
RAIO = 14


def cena(escala=1.0, manchas=MANCHAS, retrato=False):
    """Folha (elipse verde, raiz da area ~ 702 px em escala 1) sobre solo marrom, com manchas laranja dentro e uma
    fora. Devolve (imagem BGR, mascara verdadeira da folha)."""
    e = lambda v: int(round(v * escala))
    img = np.zeros((e(1080), e(1920), 3), np.uint8)
    img[:] = MARROM
    mascara = np.zeros(img.shape[:2], np.uint8)
    for alvo, cor in ((img, VERDE), (mascara, 255)):
        cv2.ellipse(alvo, (e(960), e(540)), (e(560), e(280)), 0, 0, 360, cor, -1)
    for x, y in manchas:
        cv2.circle(img, (e(x), e(y)), e(RAIO), LARANJA, -1)
    cv2.circle(img, (e(150), e(150)), e(RAIO), LARANJA, -1)  # fora da folha
    if retrato:
        img, mascara = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mascara, cv2.ROTATE_90_CLOCKWISE)
    return img, mascara > 0


def circulos_da_cena(escala=1.0, **kw):
    img, m = cena(escala, **kw)
    folha = rh.normalizar_folha(img, m)
    mapa = rh.mapa_lesao(folha)
    return folha, mapa, rh.propor_circulos(folha, mapa)


def mais_proximo(circ, x, y):
    d = np.hypot(circ[:, 0] - x, circ[:, 1] - y)
    return circ[int(d.argmin())], float(d.min())


class TestNormalizar(unittest.TestCase):
    def test_raiz_da_area_vira_lado_folha(self):
        folha = rh.normalizar_folha(*cena())
        self.assertAlmostEqual(np.sqrt(folha.mascara.sum()), rh.LADO_FOLHA, delta=0.02 * rh.LADO_FOLHA)
        self.assertEqual(folha.img.shape[:2], folha.mascara.shape)
        self.assertEqual(folha.img.dtype, np.uint8)

    def test_cena_pela_metade_da_a_mesma_folha(self):
        a, b = rh.normalizar_folha(*cena(1.0)), rh.normalizar_folha(*cena(0.5))
        self.assertLessEqual(abs(a.img.shape[0] - b.img.shape[0]), 3)
        self.assertLessEqual(abs(a.img.shape[1] - b.img.shape[1]), 3)
        self.assertAlmostEqual(b.escala, 2 * a.escala, delta=0.05)

    def test_mascara_vazia_usa_a_foto_inteira(self):
        img, _ = cena(0.5)
        folha = rh.normalizar_folha(img, np.zeros(img.shape[:2], bool))
        self.assertTrue(folha.mascara.all())
        self.assertEqual(folha.caixa, (0, 0, img.shape[1], img.shape[0]))

    def test_escala_fica_dentro_dos_limites(self):
        img = np.full((60, 80, 3), 90, np.uint8)
        folha = rh.normalizar_folha(img, np.ones((60, 80), bool))
        self.assertLessEqual(folha.escala, rh.ESCALA_MAX)


class TestMapaLesao(unittest.TestCase):
    def setUp(self):
        self.img, self.m = cena()
        self.folha = rh.normalizar_folha(self.img, self.m)
        self.mapa = rh.mapa_lesao(self.folha)

    def _norm(self, x, y):
        x0, y0 = self.folha.caixa[:2]
        return int(round((x - x0) * self.folha.escala)), int(round((y - y0) * self.folha.escala))

    def test_alto_nas_manchas_baixo_no_verde_zero_fora(self):
        self.assertEqual(self.mapa.dtype, np.uint8)
        self.assertEqual(self.mapa.shape, self.folha.mascara.shape)
        for x, y in MANCHAS:
            xn, yn = self._norm(x, y)
            self.assertGreater(self.mapa[yn - 4:yn + 5, xn - 4:xn + 5].mean(), 120)
        xn, yn = self._norm(560, 540)  # verde liso, longe das manchas
        self.assertLess(self.mapa[yn - 10:yn + 11, xn - 10:xn + 11].mean(), 15)
        self.assertEqual(int(self.mapa[~self.folha.mascara].max()), 0)

    def test_brilho_branco_nao_e_lesao(self):
        img, m = cena()
        cv2.circle(img, (560, 540), 30, (235, 240, 235), -1)  # reflexo especular: quase branco
        folha = rh.normalizar_folha(img, m)
        mapa = rh.mapa_lesao(folha)
        x0, y0 = folha.caixa[:2]
        xn, yn = int((560 - x0) * folha.escala), int((540 - y0) * folha.escala)
        self.assertLess(mapa[yn - 8:yn + 9, xn - 8:xn + 9].mean(), 40)


class TestCirculos(unittest.TestCase):
    def test_um_circulo_em_cada_mancha(self):
        folha, _, circ = circulos_da_cena()
        self.assertEqual(circ.shape[1], 3)
        self.assertEqual(circ.dtype, np.float32)
        x0, y0 = folha.caixa[:2]
        for x, y in MANCHAS:
            c, d = mais_proximo(circ, (x - x0) * folha.escala, (y - y0) * folha.escala)
            r = RAIO * folha.escala
            self.assertLess(d, r / 2)
            self.assertAlmostEqual(c[2], r, delta=0.4 * r)

    def test_nenhum_centro_fora_da_folha(self):
        folha, _, circ = circulos_da_cena()
        xi, yi = np.rint(circ[:, 0]).astype(int), np.rint(circ[:, 1]).astype(int)
        self.assertTrue(folha.mascara[yi, xi].all())
        self.assertLessEqual(len(circ), rh.MAX_CIRCULOS)

    def test_folha_sem_mancha_nao_tem_circulo(self):
        _, _, circ = circulos_da_cena(manchas=[])
        self.assertEqual(circ.shape, (0, 3))

    def test_mesmos_circulos_em_outra_resolucao(self):
        _, _, a = circulos_da_cena(1.0)
        _, _, b = circulos_da_cena(0.5)
        self.assertEqual(len(a), len(b))
        for c in a:
            self.assertLess(mais_proximo(b, c[0], c[1])[1], 6)

    def test_para_original_desfaz_a_normalizacao(self):
        folha, _, circ = circulos_da_cena()
        orig = rh.para_original(circ, folha)
        c, d = mais_proximo(orig, *MANCHAS[0])
        self.assertLess(d, 4)
        self.assertAlmostEqual(c[2], RAIO, delta=0.4 * RAIO)


class TestMascaraVegetacao(unittest.TestCase):
    def test_acha_a_folha_e_mantem_as_manchas(self):
        img, m = cena(0.5)
        v = rh.mascara_vegetacao(img)
        self.assertEqual(v.dtype, bool)
        iou = (v & m).sum() / (v | m).sum()
        self.assertGreater(iou, 0.9)
        x, y = MANCHAS[1]
        self.assertTrue(v[y // 2, x // 2])  # mancha laranja dentro da folha continua na mascara

    def test_sem_vegetacao_devolve_tudo(self):
        self.assertTrue(rh.mascara_vegetacao(np.full((200, 300, 3), 128, np.uint8)).all())


def atributo(at, nome):
    return at[:, rh.NOMES_CIRCULO.index(nome)]


class TestAtributosCirculos(unittest.TestCase):
    def setUp(self):
        self.folha, self.mapa, self.circ = circulos_da_cena()
        x0, y0 = self.folha.caixa[:2]
        e = self.folha.escala
        self.na_mancha = np.array([[(MANCHAS[0][0] - x0) * e, (MANCHAS[0][1] - y0) * e, RAIO * e]], np.float32)
        self.no_verde = np.array([[(560 - x0) * e, (540 - y0) * e, RAIO * e]], np.float32)

    def test_forma_e_valores_finitos(self):
        at = rh.atributos_circulos(self.folha, self.mapa, self.circ)
        self.assertEqual(at.shape, (len(self.circ), len(rh.NOMES_CIRCULO)))
        self.assertEqual(at.dtype, np.float32)
        self.assertTrue(np.isfinite(at).all())
        self.assertEqual(len(set(rh.NOMES_CIRCULO)), len(rh.NOMES_CIRCULO))

    def test_sem_circulos_devolve_matriz_vazia(self):
        at = rh.atributos_circulos(self.folha, self.mapa, np.zeros((0, 3), np.float32))
        self.assertEqual(at.shape, (0, len(rh.NOMES_CIRCULO)))

    def test_mancha_contrasta_com_a_folha_e_tem_apoio_do_hough(self):
        at = rh.atributos_circulos(self.folha, self.mapa, self.na_mancha)
        self.assertGreater(atributo(at, "delta_a")[0], 30)
        self.assertGreater(atributo(at, "apoio")[0], 0.6)
        self.assertGreater(atributo(at, "mapa_media")[0], 0.5)
        self.assertAlmostEqual(atributo(at, "raio")[0], RAIO * self.folha.escala, delta=0.01)

    def test_verde_liso_nao_contrasta_nem_tem_apoio(self):
        at = rh.atributos_circulos(self.folha, self.mapa, self.no_verde)
        self.assertLess(abs(atributo(at, "delta_a")[0]), 3)
        self.assertLess(atributo(at, "apoio")[0], 0.3)
        self.assertLess(atributo(at, "mapa_media")[0], 0.05)

    def test_circulo_na_beirada_da_imagem_nao_quebra(self):
        h, w = self.folha.mascara.shape
        cantos = np.array([[1, 1, 40], [w - 2, h - 2, 40], [w / 2, 2, 12]], np.float32)
        at = rh.atributos_circulos(self.folha, self.mapa, cantos)
        self.assertTrue(np.isfinite(at).all())

    def test_mancha_redonda_e_mais_circular_que_um_risco(self):
        img, m = cena(manchas=[MANCHAS[0]])
        cv2.line(img, (1070, 540), (1130, 540), LARANJA, 6)  # risco laranja, como o bronzeado do acaro nas nervuras
        folha = rh.normalizar_folha(img, m)
        mapa = rh.mapa_lesao(folha)
        x0, y0 = folha.caixa[:2]
        e = folha.escala
        circ = np.array([[(MANCHAS[0][0] - x0) * e, (MANCHAS[0][1] - y0) * e, RAIO * e],
                         [(1100 - x0) * e, (540 - y0) * e, RAIO * e]], np.float32)
        at = rh.atributos_circulos(folha, mapa, circ)
        for nome in ("blob_circularidade", "blob_alongamento"):
            self.assertGreater(atributo(at, nome)[0], atributo(at, nome)[1] + 0.2, nome)
        self.assertGreater(atributo(at, "blob_circularidade")[0], 0.7)


class TestTextura(unittest.TestCase):
    def test_lbp_de_area_lisa_cai_num_so_bin(self):
        h = rh.histograma_lbp(rh.codigos_lbp(np.full((40, 40), 90, np.uint8)), np.ones((40, 40), bool))
        self.assertEqual(h.shape, (10,))
        self.assertAlmostEqual(float(h.sum()), 1.0, places=5)
        self.assertAlmostEqual(float(h.max()), 1.0, places=5)

    def test_lbp_de_ruido_se_espalha(self):
        ruido = np.random.default_rng(0).integers(0, 255, (60, 60)).astype(np.uint8)
        h = rh.histograma_lbp(rh.codigos_lbp(ruido), np.ones((60, 60), bool))
        self.assertLess(float(h.max()), 0.6)

    def test_glcm_liso_contra_xadrez(self):
        liso = rh.glcm(np.full((30, 30), 100, np.uint8))
        xadrez = rh.glcm((np.indices((30, 30)).sum(0) % 2 * 255).astype(np.uint8))
        self.assertEqual(len(liso), 4)  # contraste, homogeneidade, energia, correlacao
        self.assertAlmostEqual(liso[0], 0.0, places=5)
        self.assertAlmostEqual(liso[1], 1.0, places=5)
        self.assertAlmostEqual(liso[2], 1.0, places=5)
        self.assertGreater(xadrez[0], 100)
        self.assertLess(xadrez[1], 0.1)


class TestAtributosFolhaEExtrair(unittest.TestCase):
    def test_atributos_da_folha(self):
        folha, mapa, _ = circulos_da_cena()
        at = rh.atributos_folha(folha, mapa, confianca=0.9, fracao_area=0.24)
        self.assertEqual(at.shape, (len(rh.NOMES_FOLHA),))
        self.assertTrue(np.isfinite(at).all())
        nome = lambda n: float(at[rh.NOMES_FOLHA.index(n)])
        self.assertAlmostEqual(nome("confianca"), 0.9, places=5)
        self.assertAlmostEqual(nome("fracao_area"), 0.24, places=5)
        area_manchas = len(MANCHAS) * np.pi * RAIO ** 2 / (np.pi * 560 * 280)
        self.assertAlmostEqual(nome("mapa_frac_050"), area_manchas, delta=0.4 * area_manchas)

    def test_folha_sem_mancha_tem_fracao_zero(self):
        folha, mapa, _ = circulos_da_cena(manchas=[])
        at = rh.atributos_folha(folha, mapa, confianca=0.9, fracao_area=0.24)
        self.assertEqual(float(at[rh.NOMES_FOLHA.index("mapa_frac_030")]), 0.0)

    def test_extrair_com_mascara(self):
        img, m = cena(0.5)
        r = rh.extrair(img, m, confianca=0.8)
        self.assertEqual(r["circulos"].shape[0], r["at_circulos"].shape[0])
        self.assertEqual(len(r["circulos"]), len(MANCHAS))
        self.assertEqual(r["at_folha"].shape, (len(rh.NOMES_FOLHA),))
        self.assertAlmostEqual(r["confianca"], 0.8)
        self.assertAlmostEqual(float(r["at_folha"][rh.NOMES_FOLHA.index("fracao_area")]), m.mean(), delta=0.01)

    def test_extrair_sem_mascara_usa_a_cor_e_confianca_zero(self):
        img, _ = cena(0.5)
        r = rh.extrair(img)
        self.assertEqual(r["confianca"], 0.0)
        self.assertEqual(len(r["circulos"]), len(MANCHAS))  # a mancha fora da folha nao entra

    def test_extrair_em_imagem_sem_folha_nao_quebra(self):
        for img in (np.full((300, 400, 3), 255, np.uint8), np.zeros((300, 400, 3), np.uint8)):
            r = rh.extrair(img)
            self.assertEqual(r["circulos"].shape, (0, 3))
            self.assertEqual(r["at_circulos"].shape, (0, len(rh.NOMES_CIRCULO)))
            self.assertTrue(np.isfinite(r["at_folha"]).all())

    def test_extrair_foto_em_retrato(self):
        img, m = cena(0.5, retrato=True)
        r = rh.extrair(img, m, confianca=0.8)
        self.assertEqual(len(r["circulos"]), len(MANCHAS))


if __name__ == "__main__":
    unittest.main()
