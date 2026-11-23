"""Testes do dataset, das transformacoes e do modelo do RoCoLe (cnn_modelo.py)."""
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from comum import CLASSES_ROCOLE
from cnn_modelo import (FolhasRocole, camada_alvo, carregar, congelar_backbone, construir_modelo,
                           registrar_divisao, transformacao, verificar_divisao)


class TestDados(unittest.TestCase):
    def test_transformacao_formato(self):
        img = Image.new("RGB", (512, 288), (40, 120, 40))
        self.assertEqual(tuple(transformacao(True)(img).shape), (3, 288, 512))
        self.assertEqual(tuple(transformacao(False)(img).shape), (3, 288, 512))

    def test_dataset_e_pickle(self):  # cache temporario com 2 imagens 512x288
        with tempfile.TemporaryDirectory() as d:
            for i in ["A", "B"]:
                Image.new("RGB", (512, 288)).save(Path(d) / f"{i}.jpg")
            meta = pd.DataFrame({"id": ["A", "B"], "classe": ["rust", "healthy"], "split": ["train", "test"]})
            ds = FolhasRocole("train", treino=True, meta=meta, cache=Path(d))
            x, y, i = ds[0]
            self.assertEqual((len(ds), tuple(x.shape), y, i),
                             (1, (3, 288, 512), CLASSES_ROCOLE.index("rust"), "A"))
            pickle.dumps(ds)


class TestModelo(unittest.TestCase):
    def test_construir_modelo(self):
        self.assertEqual(construir_modelo("resnet50", 3, pretreinado=False).fc.out_features, 3)
        self.assertEqual(construir_modelo("convnext_tiny", 3, pretreinado=False).classifier[2].out_features, 3)
        with self.assertRaises(ValueError) as ctx:
            construir_modelo("resnet18", 3, pretreinado=False)
        self.assertIn("convnext_tiny", str(ctx.exception))

    def test_camada_alvo(self):
        r = construir_modelo("resnet50", 3, pretreinado=False)
        c = construir_modelo("convnext_tiny", 3, pretreinado=False)
        self.assertIs(camada_alvo(r, "resnet50"), r.layer4)
        self.assertIs(camada_alvo(c, "convnext_tiny"), c.features[-1])

    def test_congelar_backbone(self):
        m = construir_modelo("resnet50", 3, pretreinado=False)
        congelar_backbone(m, "resnet50", True)
        treinaveis = [n for n, p in m.named_parameters() if p.requires_grad]
        self.assertEqual(sorted(treinaveis), ["fc.bias", "fc.weight"])
        congelar_backbone(m, "resnet50", False)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))

    def test_carregar_sem_pesos(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            carregar("resnet50", torch.device("cpu"), caminho=Path("nao_existe.pt"))
        self.assertIn("cnn_treinar.py", str(ctx.exception))



class TestDivisaoDoCheckpoint(unittest.TestCase):
    def _arquivos(self, d, texto_meta):
        meta = Path(d) / "metadados.csv"
        meta.write_text(texto_meta)
        pesos = Path(d) / "rocole_resnet50.pt"
        pesos.write_bytes(b"pesos")
        return meta, pesos

    def test_mesma_divisao_passa(self):
        with tempfile.TemporaryDirectory() as d:
            meta, pesos = self._arquivos(d, "id,split\nA,train\n")
            registrar_divisao(pesos, meta)
            verificar_divisao(pesos, meta)

    def test_divisao_mudou_para(self):
        with tempfile.TemporaryDirectory() as d:
            meta, pesos = self._arquivos(d, "id,split\nA,train\n")
            registrar_divisao(pesos, meta)
            meta.write_text("id,split\nA,test\n")
            with self.assertRaises(SystemExit) as ctx:
                verificar_divisao(pesos, meta)
            self.assertIn("outra divisao", str(ctx.exception))

    def test_sem_registro_para(self):
        with tempfile.TemporaryDirectory() as d:
            meta, pesos = self._arquivos(d, "id,split\nA,train\n")
            with self.assertRaises(SystemExit) as ctx:
                verificar_divisao(pesos, meta)
            self.assertIn("cnn_treinar.py", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
