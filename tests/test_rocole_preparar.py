"""Testes das regras de dados do RoCoLe (rocole_preparar.py).

Rodar da raiz: PYTHONPATH=src python -m unittest discover -s tests -v
"""
import tempfile
import unittest
import zipfile
from pathlib import Path

import imagehash
import pandas as pd

from rocole_preparar import (agrupar, componentes, dividir, ler_nome, listar_imagens, main, pares_proximos,
                             remover_quase_copias_conflitantes, resolver_copias_exatas, verificar, verificar_entrada)


class TestNomes(unittest.TestCase):
    def test_listar_imagens(self):
        nomes = ["coffee___rust/", "coffee___rust/a.jpg", "coffee___rust/b.JPG", "LEIAME.txt", "coffee___rust/c.txt"]
        self.assertEqual(listar_imagens(nomes), ["coffee___rust/a.jpg", "coffee___rust/b.JPG"])

    def test_ler_nome(self):
        self.assertEqual(ler_nome("coffee___rust/C10P10E2.jpg"),
                         {"id": "C10P10E2", "classe": "rust", "planta": "C10P10", "lado": "E"})
        self.assertEqual(ler_nome("coffee___rust/C11P11HE.jpg")["planta"], "C11P11")
        self.assertEqual(ler_nome("coffee___rust/C11P11HE.jpg")["lado"], "")
        self.assertEqual(ler_nome("coffee___healthy/foto.jpg")["planta"], "foto")  # nome fora do padrao
        self.assertEqual(ler_nome("coffee___red_spider_mite/C2P14H2.JPG"),
                         {"id": "C2P14H2", "classe": "red_spider_mite", "planta": "C2P14", "lado": "H"})


class TestCopias(unittest.TestCase):
    def test_resolver_copias_exatas(self):
        df = pd.DataFrame({"id": ["a", "b", "c", "d", "e"],
                           "arquivo": ["coffee___rust/a.jpg", "coffee___rust/b.jpg", "coffee___healthy/c.jpg",
                                       "coffee___rust/d.jpg", "coffee___healthy/e.jpg"],
                           "classe": ["rust", "rust", "healthy", "rust", "healthy"],
                           "sha256": ["h1", "h1", "h2", "h2", "h3"]})
        kept, desc = resolver_copias_exatas(df)
        self.assertEqual(kept["id"].tolist(), ["a", "e"])
        self.assertEqual(dict(desc), {"coffee___rust/b.jpg": "copia exata de a",
                                      "coffee___healthy/c.jpg": "copia exata com rotulo conflitante",
                                      "coffee___rust/d.jpg": "copia exata com rotulo conflitante"})

    def test_pares_proximos(self):
        h = [imagehash.hex_to_hash(x) for x in ["0000000000000000", "000000000000000f", "00000000000000ff"]]
        self.assertEqual(pares_proximos(h), [(0, 1), (1, 2)])
        self.assertEqual(pares_proximos(h, limiar=3), [])

    def test_componentes(self):
        self.assertEqual(componentes(5, [(0, 1), (2, 1)]), [0, 0, 0, 3, 4])

    def test_remover_quase_copias_conflitantes(self):
        df = pd.DataFrame({"id": list("abcde"), "arquivo": [f"x/{c}.jpg" for c in "abcde"],
                           "classe": ["rust", "healthy", "rust", "rust", "healthy"], "quase": [0, 0, 2, 2, 4]})
        kept, desc = remover_quase_copias_conflitantes(df)
        self.assertEqual(kept["id"].tolist(), ["c", "d", "e"])
        self.assertEqual(dict(desc), {"x/a.jpg": "quase-copia com rotulo conflitante",
                                      "x/b.jpg": "quase-copia com rotulo conflitante"})


class TestDivisao(unittest.TestCase):
    def test_agrupar(self):
        df = pd.DataFrame({"planta": ["C1P1", "C1P1", "C1P2", "C1P3"], "sha256": list("abcd")})
        g = agrupar(df, pares=[(1, 2)])
        self.assertTrue(g[0] == g[1] == g[2])
        self.assertNotEqual(g[2], g[3])

    def test_agrupar_liga_plantas_por_copia_exata(self):
        # C11P23E1 e copia exata de C12P36E1: as duas plantas sao a mesma, entao os outros lados (H1) se juntam
        df = pd.DataFrame({"planta": ["C11P23", "C12P36", "C11P23", "C12P36"], "sha256": ["x", "x", "a", "b"],
                           "quase": [0, 1, 2, 3]})
        g = agrupar(df)
        self.assertEqual(g[2], g[3])

    def test_filtrar_e_agrupar_mantem_ligacoes_de_descartadas_e_refotos(self):
        from rocole_preparar import filtrar_e_agrupar
        h = imagehash.hex_to_hash
        todos = pd.DataFrame({
            "id": ["C11P23E1", "C12P36E1", "C11P23H1", "C12P36H1", "C2P8E1", "C2P7E1", "C5P1E1"],
            "classe": ["red_spider_mite", "healthy", "healthy", "healthy", "healthy", "healthy", "rust"],
            "planta": ["C11P23", "C12P36", "C11P23", "C12P36", "C2P8", "C2P7", "C5P1"],
            "sha256": ["x", "x", "a", "b", "c", "d", "e"],
            "phash": [h("0000000000000000"), h("0000000000000000"),  # copia exata, rotulos conflitantes
                      h("ffffff0000000000"), h("ffffff000000003f"),  # distancia 6
                      h("000000ffffff0000"), h("000000ffffff03ff"),  # distancia 10 (refoto da mesma folha)
                      h("0000000000ffffff")]})
        todos["arquivo"] = todos["classe"].radd("coffee___") + "/" + todos["id"] + ".jpg"
        df, desc, _ = filtrar_e_agrupar(todos)
        self.assertEqual(sorted(a for a, _ in desc),
                         ["coffee___healthy/C12P36E1.jpg", "coffee___red_spider_mite/C11P23E1.jpg"])
        g = df.set_index("id")["grupo"]
        self.assertEqual(g["C11P23H1"], g["C12P36H1"])  # ligados pela copia descartada
        self.assertEqual(g["C2P8E1"], g["C2P7E1"])  # ligados pelo pHash 10
        self.assertNotIn(g["C5P1E1"], {g["C11P23H1"], g["C2P8E1"]})

    def _sintetico(self):  # 140 grupos de 4 imagens, classes desbalanceadas
        cl = ["healthy"] * 2 + ["rust"] * 2 + ["red_spider_mite"]
        linhas = [{"classe": cl[(g + j) % 5], "grupo": g, "sha256": f"{g}-{j}"} for g in range(140) for j in range(4)]
        return pd.DataFrame(linhas)

    def test_dividir_proporcoes_e_grupos(self):
        df = self._sintetico()
        df["split"] = dividir(df)
        frac = df["split"].value_counts(normalize=True)
        self.assertAlmostEqual(frac["train"], 0.70, delta=0.05)
        self.assertAlmostEqual(frac["val"], 0.15, delta=0.05)
        self.assertAlmostEqual(frac["test"], 0.15, delta=0.05)
        self.assertTrue((df.groupby("grupo")["split"].nunique() == 1).all())
        self.assertTrue(dividir(df).equals(df["split"]))  # deterministico
        verificar(df)

    def test_verificar_acusa_vazamento(self):
        df = self._sintetico()
        df["split"] = dividir(df)
        df.loc[0, "split"] = "test" if df.loc[0, "split"] != "test" else "train"
        with self.assertRaises(AssertionError):
            verificar(df)

    def test_verificar_acusa_quase_copia_entre_splits(self):
        h = imagehash.hex_to_hash
        df = pd.DataFrame({"grupo": [0, 1], "sha256": ["a", "b"], "split": ["train", "test"],
                           "phash": [h("0000000000000000"), h("000000000000003f")]})  # distancia 6
        with self.assertRaises(AssertionError):
            verificar(df)


class TestMain(unittest.TestCase):
    def test_main_sem_zip(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--zip", "nao_existe.zip"])
        self.assertIn("--zip", str(ctx.exception))

    def test_main_arquivo_que_nao_e_zip(self):
        with tempfile.TemporaryDirectory() as d:
            falso = Path(d) / "archive.zip"
            falso.write_text("isto nao e um zip")
            with self.assertRaises(SystemExit) as ctx:
                main(["--zip", str(falso)])
            self.assertIn("nao e um zip valido", str(ctx.exception))

    def test_main_zip_com_pasta_a_mais(self):
        with tempfile.TemporaryDirectory() as d:
            caminho = Path(d) / "archive.zip"
            with zipfile.ZipFile(caminho, "w") as z:
                z.writestr("archive/coffee___rust/C1P1H1.jpg", b"x")
            with self.assertRaises(SystemExit) as ctx:
                main(["--zip", str(caminho)])
            self.assertIn("nenhuma imagem coffee___<classe>/<arquivo>", str(ctx.exception))


class TestEntrada(unittest.TestCase):
    def test_classe_desconhecida(self):
        with self.assertRaises(SystemExit) as ctx:
            verificar_entrada(["coffee___rust/C1P1H1.jpg", "coffee___miner/C1P1E1.jpg"])
        self.assertIn("miner", str(ctx.exception))

    def test_nome_repetido_em_pastas_diferentes(self):
        with self.assertRaises(SystemExit) as ctx:
            verificar_entrada(["coffee___rust/C1P1H1.jpg", "coffee___healthy/C1P1H1.jpg"])
        self.assertIn("C1P1H1", str(ctx.exception))

    def test_entrada_valida_passa(self):
        verificar_entrada(["coffee___rust/C1P1H1.jpg", "coffee___healthy/C1P1E1.jpg"])


if __name__ == "__main__":
    unittest.main()
