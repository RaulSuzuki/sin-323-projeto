"""Testes do detector de ferrugem por regras (replica de Rodriguez-Gallo et al., 2023) em rocole_regras.py."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rocole_regras import (bootstrap_binario, carregar_parametros, circulos, contar_grade, escolher_parametros,
                           limpar_mascara, mascara_cor, metricas_binarias, taxas_por_classe_e_lado, variancia_local)

VERDE = (0, 160, 0)  # BGR
LARANJA = (0, 128, 255)


def folha_com_manchas(centros, raio=15, ruido=0):
    rng = np.random.default_rng(0)
    img = np.zeros((400, 600, 3), np.uint8)
    img[:] = VERDE
    for c in centros:
        cv2.circle(img, c, raio, LARANJA, -1)
    if ruido:
        img = np.clip(img.astype(int) + rng.integers(-ruido, ruido + 1, img.shape), 0, 255).astype(np.uint8)
    return img


def detectar(img, preset="medio", param2=18):
    return circulos(limpar_mascara(mascara_cor(variancia_local(img), preset)), param2)


class TestVariancia(unittest.TestCase):
    def test_constante_da_zero(self):
        img = np.full((50, 60, 3), 90, np.uint8)
        self.assertEqual(int(variancia_local(img).max()), 0)

    def test_igual_ao_calculo_direto(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 20, (30, 30, 3)).astype(np.uint8)  # variancia < 255, sem corte
        v = variancia_local(img)
        y, x = 15, 12
        viz = [img[y + dy, x + dx].astype(float) for dy in range(-2, 3) for dx in range(-2, 3) if dx * dx + dy * dy <= 5]
        self.assertEqual(len(viz), 21)  # janela circular de raio 2, como o filtro Variance do ImageJ
        esperado = np.var(np.array(viz), axis=0)
        np.testing.assert_allclose(v[y, x].astype(float), esperado, atol=1.0)

    def test_corta_em_255(self):
        img = np.zeros((20, 20, 3), np.uint8)
        img[:, 10:] = 255
        self.assertEqual(int(variancia_local(img).max()), 255)


class TestMascara(unittest.TestCase):
    def test_pega_vermelho_e_laranja_e_ignora_o_resto(self):
        cores = {"vermelho": (0, 0, 255), "laranja": (0, 128, 255), "verde": (0, 255, 0), "azul": (255, 0, 0),
                 "branco": (255, 255, 255), "vermelho_escuro": (0, 0, 50)}
        img = np.array([[c for c in cores.values()]], np.uint8)
        m = mascara_cor(img, "medio")[0] > 0
        self.assertEqual(dict(zip(cores, m.tolist())),
                         {"vermelho": True, "laranja": True, "verde": False, "azul": False, "branco": False,
                          "vermelho_escuro": False})


class TestDetector(unittest.TestCase):
    def test_acha_manchas_laranja_em_folha_verde(self):
        centros = [(100, 100), (300, 200), (480, 320)]
        c = detectar(folha_com_manchas(centros))
        for cx, cy in centros:
            d = np.hypot(c[:, 0] - cx, c[:, 1] - cy).min() if len(c) else np.inf
            self.assertLess(d, 6, f"mancha em {(cx, cy)} nao encontrada: {c}")
        self.assertLessEqual(len(c), 5)

    def test_folha_lisa_nao_tem_mancha(self):
        self.assertEqual(len(detectar(folha_com_manchas([], ruido=3))), 0)

    def test_mascara_da_folha_ignora_manchas_fora_dela(self):
        img = folha_com_manchas([(100, 100)])
        folha = np.zeros(img.shape[:2], bool)
        folha[200:, 300:] = True  # folha-alvo longe da mancha
        self.assertGreaterEqual(contar_grade(img)["medio_p18"], 1)
        self.assertTrue(all(n == 0 for n in contar_grade(img, mascara=folha).values()))

    def test_contar_grade_tem_uma_contagem_por_combinacao(self):
        cont = contar_grade(folha_com_manchas([(100, 100)]))
        self.assertEqual(len(cont), 9)  # 3 presets de cor x 3 sensibilidades do Hough
        self.assertTrue(all(isinstance(n, int) for n in cont.values()))


class TestAvaliacao(unittest.TestCase):
    def test_metricas_binarias(self):
        m = metricas_binarias(np.array([1, 1, 0, 0, 0]), np.array([1, 0, 1, 0, 0]))
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
        tab = bootstrap_binario(y, pred, grupos)
        self.assertAlmostEqual(tab.loc["sensibilidade", "estimativa"], 0.5)
        self.assertEqual((tab.loc["sensibilidade", "IC95_inf"], tab.loc["sensibilidade", "IC95_sup"]), (0.0, 1.0))

    def test_escolher_parametros_pega_a_melhor_combinacao(self):
        # 'medio_p18' com k=2 separa perfeitamente; as outras combinacoes erram
        cont = pd.DataFrame({"medio_p18": [5, 3, 2, 0, 1, 1], "largo_p12": [9, 9, 9, 9, 9, 9]})
        y = np.array([1, 1, 1, 0, 0, 0])
        p = escolher_parametros(cont, y, ks=(1, 2, 3))
        self.assertEqual((p["preset"], p["param2"], p["k"]), ("medio", 18, 2))
        self.assertAlmostEqual(p["acuracia_balanceada"], 1.0)

    def test_taxas_por_classe_e_lado(self):
        df = pd.DataFrame({"classe": ["rust", "rust", "rust", "healthy", "healthy", "red_spider_mite"],
                           "lado": ["H", "H", "E", "H", "E", "E"],
                           "alarme": [True, False, False, True, False, True]})
        t = taxas_por_classe_e_lado(df, "alarme")
        self.assertAlmostEqual(t.loc["rust", "H"], 0.5)
        self.assertAlmostEqual(t.loc["rust", "E"], 0.0)
        self.assertAlmostEqual(t.loc["healthy", "H"], 1.0)
        self.assertAlmostEqual(t.loc["red_spider_mite", "E"], 1.0)

    def test_carregar_parametros_sem_ajuste(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as ctx:
                carregar_parametros(Path(d) / "parametros.json")
            self.assertIn("ajustar", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
