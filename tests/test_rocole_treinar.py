"""Testes das partes puras do treino do RoCoLe (rocole_treinar.py)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rocole_treinar import pesos_classes, treinar


class Minimo(nn.Module):
    """Rede minima com cabeca 'fc', como a ResNet-50 (congelar_backbone usa o prefixo 'fc.')."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(self.conv(x).mean((2, 3)))


class FalhaNaSegundaVez:
    """Loader que funciona uma vez e depois simula uma interrupcao (OOM, Ctrl-C)."""

    def __init__(self, lotes):
        self.lotes, self.vezes = lotes, 0

    def __iter__(self):
        self.vezes += 1
        if self.vezes > 1:
            raise RuntimeError("interrompido")
        return iter(self.lotes)


class TestPesos(unittest.TestCase):
    def test_pesos_classes(self):
        w = pesos_classes(np.array([0, 0, 1, 2, 2, 2]), 3)
        np.testing.assert_allclose(w, [0.8182, 1.6364, 0.5455], atol=1e-4)
        self.assertAlmostEqual(w.sum(), 3.0)


class TestTreinar(unittest.TestCase):
    def test_historico_gravado_a_cada_epoca(self):
        torch.manual_seed(0)
        lotes = [(torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 0]), ["a", "b", "c", "d"])]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with self.assertRaises(RuntimeError):
                treinar(Minimo(), "resnet50", lotes, FalhaNaSegundaVez(lotes), nn.CrossEntropyLoss(),
                        torch.device("cpu"), epocas=3, epocas_cabeca=1, destino=d / "m.pt", historico=d / "h.csv")
            hist = pd.read_csv(d / "h.csv")
            self.assertEqual(hist["epoca"].tolist(), [1])
            self.assertTrue((d / "m.pt").exists())


if __name__ == "__main__":
    unittest.main()
