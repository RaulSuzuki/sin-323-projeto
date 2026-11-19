"""Testes da validacao cruzada de entradas (rocole_cv.py) e dos nomes por entrada (rocole_modelo.py)."""
import unittest

import pandas as pd

from rocole_cv import decidir, dobras_cv
from rocole_modelo import ENTRADAS, camada_alvo, construir_modelo, nome_execucao


def meta_sintetico():
    cl = ["healthy"] * 2 + ["rust"] * 2 + ["red_spider_mite"]
    linhas = [{"id": f"{g}-{j}", "classe": cl[(g + j) % 5], "grupo": g, "split": "test" if g >= 130 else "train"}
              for g in range(150) for j in range(4)]
    return pd.DataFrame(linhas)


class TestDobras(unittest.TestCase):
    def test_teste_original_fica_de_fora(self):
        for d in dobras_cv(meta_sintetico()):
            self.assertEqual(len(d), 130 * 4)  # so treino + validacao entram na validacao cruzada

    def test_cada_foto_e_avaliada_uma_vez_e_grupos_nao_se_dividem(self):
        dobras = dobras_cv(meta_sintetico())
        self.assertEqual(len(dobras), 3)
        fora = pd.concat([d[d["split"] == "test"] for d in dobras])
        self.assertEqual(sorted(fora["id"]), sorted(meta_sintetico().query("split != 'test'")["id"]))
        for d in dobras:
            self.assertTrue((d.groupby("grupo")["split"].nunique() == 1).all())
            self.assertEqual(set(d["split"]), {"train", "val", "test"})
            frac = d["split"].value_counts(normalize=True)
            self.assertAlmostEqual(frac["test"], 1 / 3, delta=0.05)

    def test_deterministico(self):
        a, b = dobras_cv(meta_sintetico()), dobras_cv(meta_sintetico())
        self.assertTrue(all(x["split"].equals(y["split"]) for x, y in zip(a, b)))


class TestDecisao(unittest.TestCase):
    def _res(self, foto, recorte, mascara):
        linhas = [{"entrada": e, "dobra": k, "f1_macro": v}
                  for e, vals in [("foto", foto), ("recorte", recorte), ("recorte_mascara", mascara)]
                  for k, v in enumerate(vals)]
        return pd.DataFrame(linhas)

    def test_adota_recorte_que_ganha_na_media_e_em_2_dobras(self):
        d = decidir(self._res([0.80, 0.80, 0.80], [0.85, 0.82, 0.79], [0.70, 0.70, 0.70]))
        self.assertEqual(d["escolhida"], "recorte")

    def test_media_maior_mas_so_1_dobra_nao_basta(self):
        d = decidir(self._res([0.80, 0.80, 0.80], [0.95, 0.79, 0.79], [0.70, 0.70, 0.70]))
        self.assertEqual(d["escolhida"], "foto")

    def test_se_as_duas_passam_fica_a_de_maior_media(self):
        d = decidir(self._res([0.80, 0.80, 0.80], [0.82, 0.82, 0.82], [0.85, 0.84, 0.83]))
        self.assertEqual(d["escolhida"], "recorte_mascara")


class TestNomes(unittest.TestCase):
    def test_nome_execucao(self):
        self.assertEqual(nome_execucao("resnet50", "foto"), "resnet50")  # nomes de hoje nao mudam
        self.assertEqual(nome_execucao("resnet50", "recorte"), "resnet50_recorte")
        self.assertEqual(set(ENTRADAS), {"foto", "recorte", "recorte_mascara"})
        r = construir_modelo("resnet50", 3, pretreinado=False)
        self.assertIs(camada_alvo(r, "resnet50_recorte"), r.layer4)  # nome da execucao, nao so da arquitetura


if __name__ == "__main__":
    unittest.main()
