"""Testes da escolha de mascara e do recorte da folha-alvo (rocole_folha.py). Sem SAM: so as funcoes puras."""
import unittest

import numpy as np

from rocole_folha import FUNDO, caixa, componente_do_centro, escolher_mascara, recortar


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

    def test_confianca_baixa_vira_foto_inteira(self):
        h, w = 100, 200
        m, info = escolher_mascara(retangulo(h, w, 20, 80, 50, 150)[None], np.array([0.5]))
        self.assertIsNone(m)
        self.assertFalse(info["ok"])
        self.assertEqual(info["motivo"], "baixa confianca")

    def test_sem_mascara_valida_no_centro(self):
        m, info = escolher_mascara(retangulo(100, 200, 0, 30, 0, 60)[None], np.array([0.99]))
        self.assertIsNone(m)
        self.assertEqual(info["motivo"], "sem mascara valida no centro")


class TestRecorte(unittest.TestCase):
    def test_caixa_com_margem_e_limites(self):
        m = retangulo(300, 400, 50, 150, 100, 200)  # caixa 100x100
        self.assertEqual(caixa(m, margem=0.10), (90, 40, 210, 160))
        self.assertEqual(caixa(retangulo(300, 400, 0, 100, 0, 100), margem=0.10), (0, 0, 110, 110))

    def test_recorte_tem_o_tamanho_da_rede_e_preenche_com_o_fundo(self):
        img = np.full((300, 400, 3), 200, np.uint8)
        out = recortar(img, (0, 0, 100, 300))  # recorte alto e estreito: sobra borda dos lados
        self.assertEqual(out.shape, (288, 512, 3))
        self.assertEqual(tuple(out[144, 0]), FUNDO)
        self.assertEqual(tuple(out[144, 256]), (200, 200, 200))

    def test_recorte_com_mascara_apaga_o_fundo(self):
        img = np.full((300, 400, 3), 200, np.uint8)
        m = retangulo(300, 400, 0, 300, 0, 200)  # so a metade esquerda e folha
        out = recortar(img, (0, 0, 400, 300), mascara=m)
        self.assertEqual(tuple(out[150, 100]), (200, 200, 200))
        self.assertEqual(tuple(out[150, 400]), FUNDO)


if __name__ == "__main__":
    unittest.main()
