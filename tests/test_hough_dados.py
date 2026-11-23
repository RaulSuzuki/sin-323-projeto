"""Testes de hough_dados.py: os conjuntos (rocole, bracol) que o detector de ferrugem sabe usar."""
import argparse
import unittest

import pandas as pd

import hough_dados as hd
import hough_treinar as ht
from comum import CLASSES_ROCOLE, MODELOS_ROCOLE, OUT_ROCOLE


class TestClasseBracol(unittest.TestCase):
    def test_ferrugem_vence_outra_doenca_e_sem_doenca_e_saudavel(self):
        df = pd.DataFrame({"miner": [0, 1, 1, 0, 0], "rust": [0, 0, 1, 1, 0], "phoma": [0, 0, 0, 0, 1],
                           "cercospora": [0, 0, 0, 0, 0]})
        self.assertEqual(hd.classe_bracol(df).tolist(), ["healthy", "other", "rust", "rust", "other"])


class TestMetadados(unittest.TestCase):
    def test_bracol_tem_as_colunas_do_detector_e_a_divisao_do_bracol(self):
        m = hd.meta_bracol()
        self.assertEqual(len(m), 1401)
        for col in ("id", "arquivo", "classe", "grupo", "split", "origem"):
            self.assertIn(col, m.columns)
        self.assertEqual(m["split"].value_counts().to_dict(), {"train": 1000, "test": 201, "val": 200})
        self.assertEqual(set(m["classe"]), {"healthy", "other", "rust"})
        self.assertEqual(int((m["classe"] == "healthy").sum()), 144)
        self.assertTrue(m["id"].map(lambda i: isinstance(i, str)).all())
        self.assertTrue(m["id"].is_unique)
        r = m.iloc[0]
        self.assertTrue(str(r["arquivo"]).replace("\\", "/").endswith(f"leaf/images/{r['id']}.jpg"))
        self.assertTrue((m.groupby("grupo")["split"].nunique() == 1).all())  # nenhum grupo em dois splits

    def test_rocole_e_o_metadados_de_sempre(self):
        m = hd.meta_rocole()
        ref = pd.read_csv(OUT_ROCOLE / "metadados.csv")
        self.assertEqual(m["id"].tolist(), ref["id"].tolist())
        self.assertEqual(m["classe"].tolist(), ref["classe"].tolist())
        self.assertTrue((m["origem"] == "rocole").all())

    def test_so_ha_os_dois_conjuntos_cada_um_com_o_seu_modelo(self):
        self.assertEqual(hd.CONJUNTOS, ("rocole", "bracol"))
        self.assertFalse(hasattr(hd, "meta_ambos"))
        with self.assertRaises(SystemExit):
            hd.carregar_conjunto("ambos")


class TestConjunto(unittest.TestCase):
    def test_rocole_guarda_tudo_onde_ja_estava(self):
        c = hd.carregar_conjunto("rocole")
        self.assertEqual(c.classes, CLASSES_ROCOLE)
        self.assertEqual(c.pasta, ht.PASTA)
        self.assertEqual(c.artefato, ht.ARTEFATO)
        self.assertEqual(c.cache(60, 12), ht.arquivo_cache(60, 12))
        self.assertEqual([str(p) for p in c.pastas_mascaras], [str(p) for p in ht.pastas_mascaras()])

    def test_bracol_tem_pasta_e_modelo_proprios(self):
        b = hd.carregar_conjunto("bracol")
        self.assertEqual(b.pasta, OUT_ROCOLE / "hough_bracol")
        self.assertEqual(b.artefato, MODELOS_ROCOLE / "bracol_hough.joblib")
        self.assertEqual(b.classes, ["healthy", "other", "rust"])
        self.assertNotEqual(b.cache(60, 12), hd.carregar_conjunto("rocole").cache(60, 12))
        self.assertNotEqual(b.sha256(), hd.carregar_conjunto("rocole").sha256())

    def test_conjunto_desconhecido_para_com_mensagem(self):
        with self.assertRaises(SystemExit):
            hd.carregar_conjunto("kenya")


class TestOpcaoDataset(unittest.TestCase):
    def _parser(self):
        ap = argparse.ArgumentParser()
        hd.adicionar_opcao_dataset(ap)
        return ap

    def test_padrao_e_rocole(self):
        self.assertEqual(self._parser().parse_args([]).dataset, "rocole")

    def test_dataset_e_atalhos_dao_no_mesmo(self):
        ap = self._parser()
        self.assertEqual(ap.parse_args(["--dataset", "bracol"]).dataset, "bracol")
        self.assertEqual(ap.parse_args(["--bracol"]).dataset, "bracol")
        self.assertEqual(ap.parse_args(["--rocole"]).dataset, "rocole")
        with self.assertRaises(SystemExit):
            ap.parse_args(["--ambos"])


class TestDobrasBracol(unittest.TestCase):
    def test_tres_dobras_sem_grupo_dividido_e_com_ferrugem_em_todas(self):
        m = hd.meta_bracol()
        d = ht.dobras(m)
        tv = m[m["split"].isin(["train", "val"])].set_index("id").loc[d.index]
        self.assertEqual(sorted(d.unique()), [0, 1, 2])
        self.assertTrue((pd.crosstab(d.to_numpy(), tv["classe"].to_numpy())["rust"] > 100).all())
        self.assertTrue((pd.DataFrame({"d": d.to_numpy(), "g": tv["grupo"].to_numpy()}).groupby("g")["d"].nunique() == 1).all())


if __name__ == "__main__":
    unittest.main()
