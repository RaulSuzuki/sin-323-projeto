"""Testes da escolha da mascara da folha (folha_sam.py). Sem SAM: so as funcoes puras."""
import unittest

import numpy as np

from folha_sam import componente_do_centro, escolher_mascara


def retangulo(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


class TestMascara(unittest.TestCase):
    def test_componente_do_centro_descarta_pedacos_soltos(self):
        m = retangulo(100, 200, 30, 70, 60, 140) | retangulo(100, 200, 0, 10, 0, 10)
        c = componente_do_centro(m, (100, 50))
        self.assertEqual(int(c.sum()), 40 * 80)
        self.assertFalse(c[5, 5])

    def test_componente_do_centro_vazia_se_o_centro_esta_fora(self):
        self.assertEqual(int(componente_do_centro(retangulo(100, 200, 0, 10, 0, 10), (100, 50)).sum()), 0)

    def test_escolhe_a_de_maior_confianca_entre_as_validas(self):
        h, w = 100, 200
        mascaras = np.stack([retangulo(h, w, 48, 52, 98, 102),  # cobre o centro, mas pequena demais (0,08%)
                             retangulo(h, w, 20, 80, 50, 150),  # valida, area 30%
                             retangulo(h, w, 0, 30, 0, 60)])  # maior confianca, mas fora do centro
        m, info = escolher_mascara(mascaras, np.array([0.95, 0.90, 0.99]))
        self.assertTrue(info["ok"])
        self.assertEqual(info["indice"], 1)
        self.assertAlmostEqual(info["area"], 0.30)
        self.assertTrue(m[50, 100])

    def test_confianca_baixa_fica_marcada(self):
        h, w = 100, 200
        m, info = escolher_mascara(retangulo(h, w, 20, 80, 50, 150)[None], np.array([0.5]))
        self.assertIsNone(m)
        self.assertFalse(info["ok"])
        self.assertEqual(info["motivo"], "baixa confianca")

    def test_sem_mascara_valida_no_centro(self):
        m, info = escolher_mascara(retangulo(100, 200, 0, 30, 0, 60)[None], np.array([0.99]))
        self.assertIsNone(m)
        self.assertEqual(info["motivo"], "sem mascara valida no centro")


if __name__ == "__main__":
    unittest.main()
