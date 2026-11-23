"""Testes da opcao de recusa (hough_recusa.py): faixa em volta do limiar, curva aninhada e escolha da taxa."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import hough_recusa as hr


class TestFaixa(unittest.TestCase):
    def test_margem_e_o_quantil_da_distancia_ao_limiar(self):
        p = np.array([0.30, 0.35, 0.45, 0.70, 0.90, 0.10, 0.05, 0.20, 0.60, 0.50])  # distancias a 0,30: 0 a 0,60
        self.assertEqual(hr.margem_para_taxa(p, 0.30, 0.0), 0.0)
        m = hr.margem_para_taxa(p, 0.30, 0.30)
        self.assertEqual(int((np.abs(p - 0.30) < m).sum()), 3)  # recusa 30% das 10 fotos

    def test_decidir_recusa_so_dentro_da_faixa(self):
        d = hr.decidir(np.array([0.05, 0.25, 0.30, 0.34, 0.36, 0.90]), limiar=0.30, margem=0.05)
        self.assertEqual(d.tolist(), [0, -1, -1, -1, 1, 1])
        self.assertEqual(hr.decidir(np.array([0.29, 0.30]), 0.30, 0.0).tolist(), [0, 1])  # sem margem, ninguem e recusado

    def test_metricas_so_nas_fotos_aceitas(self):
        y = np.array([1, 1, 1, 0, 0, 0], bool)
        p = np.array([0.90, 0.32, 0.10, 0.80, 0.28, 0.05])
        m = hr.metricas_com_recusa(y, p, limiar=0.30, margem=0.05)  # recusa 0,32 (ferrugem) e 0,28 (sem ferrugem)
        self.assertAlmostEqual(m["aceitas"], 4 / 6, places=6)
        self.assertEqual((m["vp"], m["fn"], m["fp"], m["vn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["f1"], 0.5, places=6)
        self.assertAlmostEqual(m["recusadas_com_ferrugem"], 0.5, places=6)
        sem = hr.metricas_com_recusa(y, p, limiar=0.30, margem=0.0)
        self.assertEqual(sem["aceitas"], 1.0)
        self.assertEqual((sem["vp"], sem["fn"], sem["fp"]), (2, 1, 1))


def oof_de_brinquedo():
    """3 dobras iguais: 4 ferrugens (0,9 0,8 0,7 e uma dificil a 0,45) e 4 sem ferrugem (0,05 0,1 0,15 e uma dificil
    a 0,55). O melhor limiar e 0,44 (F1 0,889: a saudavel a 0,55 vira alarme); recusando 25% (as 2 fotos mais perto
    do limiar, que sao as dificeis) o F1 vai a 1."""
    linhas = []
    for k in range(3):
        for i, (c, p) in enumerate([("rust", 0.9), ("rust", 0.8), ("rust", 0.7), ("rust", 0.45), ("healthy", 0.05),
                                    ("healthy", 0.1), ("healthy", 0.15), ("healthy", 0.55)]):
            linhas.append({"id": f"d{k}f{i}", "dobra": k, "classe": c, "grupo": k * 10 + i, "p_rust": p})
    return pd.DataFrame(linhas)


class TestCurvaEEscolha(unittest.TestCase):
    def test_curva_aninhada_melhora_com_a_recusa(self):
        curva = hr.curva_aninhada(oof_de_brinquedo(), taxas=(0.0, 0.25))
        self.assertEqual(sorted(curva["taxa"].unique()), [0.0, 0.25])
        self.assertEqual(len(curva), 2 * 3)
        sem, com = curva[curva["taxa"] == 0.0], curva[curva["taxa"] == 0.25]
        self.assertTrue((sem["aceitas"] == 1.0).all())
        self.assertTrue((sem["f1"] < 0.9).all())
        np.testing.assert_allclose(com["aceitas"], 0.75)
        np.testing.assert_allclose(com["f1"], 1.0)

    def test_limiar_e_margem_de_cada_dobra_vem_das_outras(self):
        oof = oof_de_brinquedo()
        oof.loc[oof["dobra"] == 0, "p_rust"] = 0.99  # a dobra 0 nao tem como escolher nada: tudo igual
        curva = hr.curva_aninhada(oof, taxas=(0.25,))
        d0 = curva[curva["dobra"] == 0].iloc[0]
        self.assertEqual(d0["aceitas"], 1.0)  # a faixa veio das dobras 1 e 2, longe de 0,99
        self.assertAlmostEqual(d0["f1"], 2 * 4 / (2 * 4 + 4), places=6)  # tudo "ferrugem": 4 certas, 4 alarmes

    def test_resumo_e_escolha_da_menor_taxa_que_bate_a_meta(self):
        curva = pd.DataFrame({"taxa": [0.0] * 3 + [0.1] * 3 + [0.2] * 3, "dobra": [0, 1, 2] * 3,
                              "aceitas": [1.0] * 3 + [0.9] * 3 + [0.8] * 3, "f1": [0.85] * 3 + [0.91, 0.90, 0.89] + [0.95] * 3})
        resumo = hr.resumo_curva(curva)
        self.assertAlmostEqual(resumo.set_index("taxa").loc[0.1, "f1_media"], 0.90, places=6)
        self.assertAlmostEqual(resumo.set_index("taxa").loc[0.1, "f1_min"], 0.89, places=6)
        self.assertEqual(hr.escolher_taxa(resumo, meta=0.90), 0.1)
        self.assertEqual(hr.escolher_taxa(resumo, meta=0.93), 0.2)
        self.assertIsNone(hr.escolher_taxa(resumo, meta=0.99))


class TestBootstrapETrava(unittest.TestCase):
    def test_ic_por_planta_contem_a_estimativa(self):
        oof = oof_de_brinquedo()
        y, p, g = (oof["classe"] == "rust").to_numpy(), oof["p_rust"].to_numpy(), oof["grupo"].to_numpy()
        ic = hr.bootstrap_recusa(y, p, g, limiar=0.44, margem=0.12, b=200)  # recusa 0,45 e 0,55
        self.assertEqual(sorted(ic.index), ["aceitas", "f1"])
        for nome in ic.index:
            self.assertLessEqual(ic.loc[nome, "IC95_inf"], ic.loc[nome, "estimativa"] + 1e-9)
            self.assertGreaterEqual(ic.loc[nome, "IC95_sup"], ic.loc[nome, "estimativa"] - 1e-9)
        self.assertAlmostEqual(ic.loc["aceitas", "estimativa"], 0.75, places=6)

    def test_teste_ja_pontuado_para(self):
        with tempfile.TemporaryDirectory() as tmp:
            arq = Path(tmp) / "recusa_teste.json"
            hr.conferir_primeira_vez(arq)
            arq.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                hr.conferir_primeira_vez(arq)
            hr.conferir_primeira_vez(arq, refazer=True)


if __name__ == "__main__":
    unittest.main()
