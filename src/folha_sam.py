"""Contorno da folha com o SAM (Segment Anything, Meta), para o treino (hough_treinar.py mascaras) e o aplicativo
(hough_inferir.py).

- Prompt: um ponto no centro da foto (a folha fica no meio). O SAM devolve 3 mascaras; fica so o pedaco de cada uma que
  contem o centro, e vale a de maior confianca com area entre 4% e 60% da foto.
- escolher_mascara marca como duvidosa a mascara com confianca < 0,75; o modelo de Hough a usa mesmo assim
  (hough.escolher_mascara_hough), porque um contorno imperfeito vale mais que a foto inteira.
Requer o modelo facebook/sam-vit-base (Hugging Face, ~375 MB, baixado na primeira vez).
"""
import cv2
import numpy as np
from PIL import Image

AREA_MIN, AREA_MAX, CONFIANCA_MIN = 0.04, 0.60, 0.75
MODELO_SAM = "facebook/sam-vit-base"


def componente_do_centro(mascara, centro):
    """So o pedaco conexo da mascara que contem o centro (cx, cy); vazio se o centro esta fora."""
    cx, cy = centro
    if not mascara[cy, cx]:
        return np.zeros_like(mascara, bool)
    _, rotulos = cv2.connectedComponents(mascara.astype(np.uint8), connectivity=8)
    return rotulos == rotulos[cy, cx]


def escolher_mascara(mascaras, confiancas, centro=None):
    """mascaras: (N, H, W) bool do SAM; confiancas: (N,). Devolve (mascara ou None, info)."""
    h, w = mascaras.shape[1:]
    centro = centro or (w // 2, h // 2)
    melhor = None
    for k in range(len(mascaras)):
        comp = componente_do_centro(mascaras[k], centro)
        area = float(comp.mean())
        if AREA_MIN <= area <= AREA_MAX and (melhor is None or confiancas[k] > melhor[2]):
            melhor = (k, comp, float(confiancas[k]), area)
    if melhor is None:
        return None, {"ok": False, "motivo": "sem mascara valida no centro", "indice": -1, "area": np.nan,
                      "confianca": np.nan}
    k, comp, conf, area = melhor
    info = {"ok": conf >= CONFIANCA_MIN, "motivo": "" if conf >= CONFIANCA_MIN else "baixa confianca",
            "indice": k, "area": area, "confianca": conf}
    return (comp if info["ok"] else None), info


class SegmentadorSAM:
    def __init__(self, disp):
        import torch
        from transformers import SamModel, SamProcessor
        self.torch, self.disp = torch, disp
        self.proc = SamProcessor.from_pretrained(MODELO_SAM)
        self.modelo = SamModel.from_pretrained(MODELO_SAM).to(disp).eval()

    def __call__(self, img_rgb):
        """Tres mascaras (3, H, W) bool e suas confiancas para um ponto no centro da foto."""
        h, w = img_rgb.shape[:2]
        inp = self.proc(images=Image.fromarray(img_rgb), input_points=[[[[w / 2, h / 2]]]], return_tensors="pt")
        with self.torch.no_grad():
            out = self.modelo(**{k: v.to(self.disp) for k, v in inp.items() if k != "reshaped_input_sizes"})
        m = self.proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"], inp["reshaped_input_sizes"])
        return m[0][0].numpy().astype(bool), out.iou_scores[0, 0].float().cpu().numpy()
