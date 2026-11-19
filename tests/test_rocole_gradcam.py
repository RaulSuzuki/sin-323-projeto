"""Testes do Grad-CAM generico (gradcam_camada), com pesos aleatorios na CPU."""
import unittest

import numpy as np
import torch
import torchvision

from rocole_gradcam import gradcam_camada


class TestGradcamCamada(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(1, 3, 288, 512)

    def test_resnet50(self):
        m = torchvision.models.resnet50(weights=None, num_classes=3)
        cam, c = gradcam_camada(m, m.layer4, self.x)
        self.assertEqual(cam.shape, (288, 512))
        self.assertTrue(cam.min() >= 0 and cam.max() <= 1 and np.isfinite(cam).all())
        with torch.no_grad():
            self.assertEqual(c, int(m(self.x).argmax(1)))
        self.assertEqual(len(m.layer4._forward_hooks), 0)

    def test_convnext_tiny_classe_explicita(self):
        m = torchvision.models.convnext_tiny(weights=None, num_classes=3)
        cam, c = gradcam_camada(m, m.features[-1], self.x, classe=2)
        self.assertEqual((cam.shape, c), ((288, 512), 2))

    def test_dentro_de_no_grad(self):
        m = torchvision.models.resnet50(weights=None, num_classes=3)
        with torch.no_grad():
            cam, _ = gradcam_camada(m, m.layer4, self.x)
        self.assertEqual(cam.shape, (288, 512))


if __name__ == "__main__":
    unittest.main()
