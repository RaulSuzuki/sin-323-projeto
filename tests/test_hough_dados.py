"""Testes de hough_dados.py: o conjunto (BRACOL) que o modelo de Hough usa."""
import argparse
import unittest

import pandas as pd

import hough_dados as hd
import hough_treinar as ht
from comum import CACHE_HOUGH, MODELOS, SAIDAS


class TestClasseBracol(unittest.TestCase):
    def test_qualquer_doenca_e_doenca_e_nenhuma_doenca_vale_mais_que_outra(self):
        df = pd.DataFrame({"miner": [0, 1, 1, 0, 0, 0], "rust": [0, 0, 1, 1, 0, 0], "phoma": [0, 0, 0, 0, 1, 0],
                           "cercospora": [0, 0, 0, 0, 0, 1]})
        self.assertEqual(hd.classe_bracol(df).tolist(), ["healthy"] + ["disease"] * 5)

    def test_estrato_e_a_combinacao_das_doencas(self):
        df = pd.DataFrame({"miner": [0, 1], "rust": [0, 1], "phoma": [0, 0], "cercospora": [0, 1]})
        self.assertEqual(hd.estrato_bracol(df).tolist(), ["0000", "1101"])


class TestMetadados(unittest.TestCase):
    def test_bracol_tem_as_colunas_do_detector_e_a_divisao_do_bracol(self):
        m = hd.meta_bracol()
        self.assertEqual(len(m), 1401)
        for col in ("id", "arquivo", "classe", "grupo", "split", "origem"):
            self.assertIn(col, m.columns)
        self.assertEqual(m["split"].value_counts().to_dict(), {"train": 1000, "test": 201, "val": 200})
        self.assertEqual(set(m["classe"]), {"healthy", "disease"})
        self.assertIn("estrato", m.columns)
        self.assertEqual(int((m["classe"] == "healthy").sum()), 144)
        self.assertTrue(m["id"].map(lambda i: isinstance(i, str)).all())
        self.assertTrue(m["id"].is_unique)
        r = m.iloc[0]
        self.assertTrue(str(r["arquivo"]).replace("\\", "/").endswith(f"leaf/images/{r['id']}.jpg"))
        self.assertTrue((m.groupby("grupo")["split"].nunique() == 1).all())  # nenhum grupo em dois splits

    def test_so_ha_o_bracol(self):
        self.assertEqual(hd.CONJUNTOS, ("bracol",))
        with self.assertRaises(SystemExit):
            hd.carregar_conjunto("ambos")


class TestConjunto(unittest.TestCase):
    def test_bracol_tem_pasta_e_modelo_proprios(self):
        b = hd.carregar_conjunto("bracol")
        self.assertEqual(b.pasta, SAIDAS / "hough_bracol")
        self.assertEqual(b.artefato, MODELOS / "bracol_hough.joblib")
        self.assertEqual(b.artefato, hd.ARTEFATO_HOUGH)
        self.assertEqual(b.classes, ["healthy", "disease"])
        self.assertEqual(b.positiva, "disease")
        self.assertEqual(b.cache(60, 12), CACHE_HOUGH / "bracol_atributos_p60_12.joblib")
        self.assertEqual(b.pastas_mascaras, [hd.MASCARAS_BRACOL])
        self.assertEqual(len(b.sha256()), 64)

    def test_padrao_e_o_bracol(self):
        self.assertEqual(hd.carregar_conjunto().nome, "bracol")
        self.assertEqual((ht.CLASSES, ht.POSITIVA), (hd.CLASSES_BRACOL, "disease"))

    def test_conjunto_desconhecido_para_com_mensagem(self):
        with self.assertRaises(SystemExit):
            hd.carregar_conjunto("kenya")


class TestOpcaoDataset(unittest.TestCase):
    def _parser(self):
        ap = argparse.ArgumentParser()
        hd.adicionar_opcao_dataset(ap)
        return ap

    def test_padrao_e_bracol(self):
        self.assertEqual(self._parser().parse_args([]).dataset, "bracol")

    def test_dataset_e_atalho_dao_no_mesmo(self):
        ap = self._parser()
        self.assertEqual(ap.parse_args(["--dataset", "bracol"]).dataset, "bracol")
        self.assertEqual(ap.parse_args(["--bracol"]).dataset, "bracol")
        with self.assertRaises(SystemExit):
            ap.parse_args(["--dataset", "outro"])


class TestDobrasBracol(unittest.TestCase):
    def test_tres_dobras_sem_grupo_dividido_e_com_cada_doenca_em_todas(self):
        m = hd.meta_bracol()
        d = ht.dobras(m)
        tv = m[m["split"].isin(["train", "val"])].set_index("id").loc[d.index]
        self.assertEqual(sorted(d.unique()), [0, 1, 2])
        por_dobra = tv[hd.DOENCAS_BRACOL].groupby(d.to_numpy()).sum()
        self.assertTrue((por_dobra > 40).all().all(), por_dobra)  # cada doenca bem representada em cada dobra
        self.assertTrue((pd.crosstab(d.to_numpy(), tv["classe"].to_numpy())["healthy"] > 30).all())
        self.assertTrue((pd.DataFrame({"d": d.to_numpy(), "g": tv["grupo"].to_numpy()}).groupby("g")["d"].nunique() == 1).all())


if __name__ == "__main__":
    unittest.main()
