"""Testes dos ajustes avaliados so na validacao (cnn_validacao.py): deslocamentos por classe e folha inteira."""
import unittest

import numpy as np
import pandas as pd

from cnn_validacao import (aplicar_deslocamentos, avaliar_folhas, deslocamentos_cv, escolher_deslocamentos, folha_id,
                              folhas_com_ic, verificar_ids_teste)

# classes: 0 = healthy, 1 = red_spider_mite, 2 = rust


class TestDeslocamentos(unittest.TestCase):
    def test_aplicar_deslocamentos(self):
        p = np.array([[0.5, 0.3, 0.2]])
        self.assertEqual(aplicar_deslocamentos(p, np.zeros(3)).tolist(), [0])
        self.assertEqual(aplicar_deslocamentos(p, np.array([0.0, 1.0, 0.0])).tolist(), [1])

    def test_escolher_corrige_classe_subestimada(self):
        # o acaro sai sempre em segundo lugar, por pouco: um deslocamento pequeno no acaro acerta tudo
        p = np.array([[0.7, 0.2, 0.1]] * 2 + [[0.5, 0.4, 0.1]] * 2 + [[0.1, 0.2, 0.7]] * 2)
        y = np.array([0, 0, 1, 1, 2, 2])
        d = escolher_deslocamentos(p, y)
        self.assertEqual(d.tolist(), [0.0, 0.25, 0.0])  # o menor deslocamento que basta
        self.assertTrue((aplicar_deslocamentos(p, d) == y).all())

    def test_cv_com_modelo_perfeito_nao_muda_nada(self):
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1, 2], 20)
        p = np.full((60, 3), 0.1)
        p[np.arange(60), y] = 0.8
        grupos = rng.permutation(np.repeat(np.arange(15), 4))
        r = deslocamentos_cv(p, y, grupos, n_dobras=3)
        self.assertEqual(len(r["pred_fora_da_dobra"]), 60)
        self.assertAlmostEqual(r["f1_macro_argmax"], 1.0)
        self.assertAlmostEqual(r["f1_macro_ajustado"], 1.0)


class TestFolhas(unittest.TestCase):
    def test_folha_id(self):
        self.assertEqual(folha_id("C10P10E2"), "C10P10F2")
        self.assertEqual(folha_id("C10P10H2"), "C10P10F2")
        self.assertIsNone(folha_id("C11P11HE"))

    def test_duas_fotos_acham_ferrugem_que_so_aparece_no_verso(self):
        df = pd.DataFrame({
            "id": ["C1P1H1", "C1P1E1", "C1P2H1", "C1P2E1"],
            "classe": ["healthy", "rust", "healthy", "healthy"],
            "classe_predita": ["healthy", "rust", "healthy", "healthy"]})
        t = avaliar_folhas(df).set_index(["estrategia", "doenca"])
        self.assertAlmostEqual(t.loc[("so_face", "rust"), "sensibilidade"], 0.0)
        self.assertAlmostEqual(t.loc[("duas_fotos", "rust"), "sensibilidade"], 1.0)
        self.assertAlmostEqual(t.loc[("duas_fotos", "rust"), "especificidade"], 1.0)
        self.assertEqual(int(t.loc[("duas_fotos", "rust"), "folhas"]), 2)


def duas_plantas():
    """Planta C1P1 (grupo 0): folha 1 com ferrugem achada so no verso, folha 2 saudavel.
    Planta C1P2 (grupo 1): folha 1 com ferrugem nao achada, folha 2 saudavel."""
    ids = ["C1P1H1", "C1P1E1", "C1P1H2", "C1P1E2", "C1P2H1", "C1P2E1", "C1P2H2", "C1P2E2"]
    df = pd.DataFrame({"id": ids,
                       "classe": ["rust", "rust", "healthy", "healthy", "rust", "rust", "healthy", "healthy"],
                       "classe_predita": ["healthy", "rust"] + ["healthy"] * 6})
    meta = pd.DataFrame({"id": ids, "grupo": [0] * 4 + [1] * 4, "split": "test"})
    return df, meta


class TestFolhasNoTeste(unittest.TestCase):
    def test_ic_por_planta(self):
        df, meta = duas_plantas()
        t = folhas_com_ic(df, meta).set_index(["estrategia", "doenca", "metrica"])
        sens = t.loc[("duas_fotos", "rust", "sensibilidade")]
        self.assertAlmostEqual(sens["estimativa"], 0.5)
        self.assertEqual((sens["IC95_inf"], sens["IC95_sup"]), (0.0, 1.0))  # 2 plantas: 0, 1/2 ou 1
        espec = t.loc[("duas_fotos", "rust", "especificidade")]
        self.assertEqual((espec["estimativa"], espec["IC95_inf"], espec["IC95_sup"]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(t.loc[("so_face", "rust", "sensibilidade"), "estimativa"], 0.0)

    def test_ic_bate_com_avaliar_folhas(self):
        df, meta = duas_plantas()
        ic = folhas_com_ic(df, meta).set_index(["estrategia", "doenca", "metrica"])["estimativa"]
        for _, r in avaliar_folhas(df).iterrows():
            for m in ["sensibilidade", "especificidade"]:
                a, b = r[m], ic[(r["estrategia"], r["doenca"], m)]
                self.assertTrue((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12, (r["estrategia"], m))

    def test_predicoes_de_outra_divisao_sao_recusadas(self):
        df, meta = duas_plantas()
        verificar_ids_teste(df["id"], meta)  # mesma divisao: passa
        meta.loc[0, "split"] = "train"
        with self.assertRaises(SystemExit) as ctx:
            verificar_ids_teste(df["id"], meta)
        self.assertIn("cnn_avaliar.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
