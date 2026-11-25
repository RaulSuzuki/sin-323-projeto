"""Testes de convnext_bracol.py sem treino de verdade e sem baixar pesos: entrada, forma da rede, divisao da validacao
interna e as predicoes fora da dobra."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import convnext_bracol as cb


def meta_sintetica(n_grupos=42):
    """Folhas com as colunas do BRACOL: duas por grupo, doencas em rodizio."""
    linhas = []
    for g in range(n_grupos):
        y = [int(g % 5 == j + 1) for j in range(4)]
        for lado in range(2):
            linhas.append({"id": f"{g}_{lado}", "grupo": g, "split": "train" if g < 30 else "val",
                           **dict(zip(cb.DOENCAS, y)), "severidade": int(sum(y) > 0),
                           "estrato": "".join(map(str, y)), "classe": "disease" if sum(y) else "healthy"})
    return pd.DataFrame(linhas)


class TestEntrada(unittest.TestCase):
    def test_sempre_512x256_deitada(self):
        for forma in ((1024, 2048, 3), (2048, 1024, 3), (300, 400, 3), (90, 1000, 3)):
            self.assertEqual(cb.entrada(np.zeros(forma, np.uint8)).size, (512, 256), forma)

    def test_corta_o_centro(self):
        img = np.zeros((100, 400, 3), np.uint8)  # 4:1 -> corta as laterais
        img[:, 150:250] = (0, 0, 255)  # vermelho so no meio do recorte (100 a 300)
        saida = np.asarray(cb.entrada(img))
        self.assertGreater(int(saida[128, 256, 0]), 200)  # RGB: vermelho
        self.assertLess(int(saida[128, 5, 0]), 60)

    def test_cache_grava_so_as_que_faltam(self):
        with tempfile.TemporaryDirectory() as tmp:
            arq = Path(tmp) / "f.jpg"
            cv2.imwrite(str(arq), np.full((100, 200, 3), 128, np.uint8))
            cache = Path(tmp) / "cache"
            cb.preparar_cache(pd.DataFrame({"id": ["7"], "arquivo": [str(arq)]}), cache)
            self.assertTrue((cache / "7.jpg").exists())
            arq.unlink()  # ja esta no cache: nao precisa da original
            cb.preparar_cache(pd.DataFrame({"id": ["7"], "arquivo": [str(arq)]}), cache)


class TestRede(unittest.TestCase):
    def test_duas_cabecas(self):
        import torch
        m = cb.ModeloFolha(pretreinado=False).eval()
        with torch.no_grad():
            d, s = m(torch.zeros(2, 3, 256, 512))
        self.assertEqual(tuple(d.shape), (2, 4))
        self.assertEqual(tuple(s.shape), (2, 5))

    def test_sem_pesos_o_previsor_e_none(self):
        self.assertIsNone(cb.previsor(Path(tempfile.gettempdir()) / "nao_existe_convnext.pt"))


class TestDobras(unittest.TestCase):
    def test_validacao_interna_sem_dividir_grupos(self):
        base = meta_sintetica()
        tr, va = cb.validacao_interna(base)
        self.assertEqual(len(tr) + len(va), len(base))
        self.assertFalse(set(tr["grupo"]) & set(va["grupo"]))
        self.assertTrue(0.08 < len(va) / len(base) < 0.25)

    def test_cada_folha_prevista_so_pela_rede_que_nao_a_viu(self):
        meta = meta_sintetica()

        class Conj:
            def __init__(self, pasta):
                self.meta, self.pasta = meta, Path(pasta)

        treinadas = []

        def treinar_falso(tr, va, destino):
            treinadas.append((set(tr["id"]), set(va["id"]), destino))
            return pd.DataFrame({"epoca": [1], "map_validacao_interna": [0.5]})

        def prever_falso(destino, fora):
            k = int(Path(destino).stem[-1])
            vistas = treinadas[k][0] | treinadas[k][1]
            self.assertFalse(vistas & set(fora["id"]))  # a rede da dobra k nunca viu as folhas de fora
            return np.full((len(fora), 4), k / 10)

        original = (cb.preparar_cache, cb.PASTA_DOBRAS)
        with tempfile.TemporaryDirectory() as tmp:
            cb.preparar_cache, cb.PASTA_DOBRAS = (lambda *a, **k: None), Path(tmp) / "cv"
            try:
                oof = cb.rodar_cv(Conj(tmp), treinar_fn=treinar_falso, prever_fn=prever_falso)
            finally:
                cb.preparar_cache, cb.PASTA_DOBRAS = original
            self.assertTrue((Path(tmp) / "convnext_oof.csv").exists())
        self.assertEqual(len(treinadas), 3)
        self.assertEqual(sorted(oof["id"]), sorted(meta["id"]))
        self.assertEqual(sorted(oof["dobra"].unique()), [0, 1, 2])
        np.testing.assert_allclose(oof["p_rust"], oof["dobra"] / 10)


if __name__ == "__main__":
    unittest.main()
