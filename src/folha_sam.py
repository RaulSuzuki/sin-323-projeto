"""Segmenta a folha-alvo do RoCoLe com o SAM (Segment Anything, Meta) e gera duas entradas recortadas para a CNN.

Uso: python -P src/folha_sam.py [--zip data/archive.zip]

- Prompt: um ponto no centro da foto (a folha-alvo esta quase sempre no meio). O SAM devolve 3 mascaras; fica so o
  pedaco de cada uma que contem o centro, e vale a de maior confianca com area entre 4% e 60% da foto.
- Mascara duvidosa (centro fora da folha, area fora da faixa, confianca < 0,75): a foto entra inteira, como hoje.
  Assim um recorte errado nunca deixa a entrada pior do que a foto inteira.
- Saidas: data/cache_rocole_recorte_512x288 (caixa da folha + 10% de margem, com o fundo em volta),
  data/cache_rocole_recorte_mascara_512x288 (a mesma caixa, com o fundo apagado), data/mascaras_rocole/<id>.png
  (mascara em 960x540) e outputs/folhas_sam.csv (area, confianca, caixa, motivo da recusa).
Requer o modelo facebook/sam-vit-base (Hugging Face, ~375 MB, baixado na primeira vez).
"""
import argparse
import io
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from comum import (OUT_ROCOLE, ROCOLE_ALTURA, ROCOLE_CACHE_RECORTE, ROCOLE_CACHE_RECORTE_MASCARA, ROCOLE_LARGURA,
                   ROCOLE_MASCARAS, ROCOLE_ZIP, SEED)

FUNDO = (124, 116, 104)  # media do ImageNet em RGB: depois da normalizacao vira ~0 (neutro)
AREA_MIN, AREA_MAX, CONFIANCA_MIN, MARGEM = 0.04, 0.60, 0.75, 0.10
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


def caixa(mascara, margem=MARGEM):
    """(x0, y0, x1, y1) da mascara, com margem proporcional ao tamanho da caixa, dentro da imagem."""
    ys, xs = np.nonzero(mascara)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    dx, dy = round(margem * (x1 - x0)), round(margem * (y1 - y0))
    h, w = mascara.shape
    return max(0, x0 - dx), max(0, y0 - dy), min(w, x1 + dx), min(h, y1 + dy)


def recortar(img_rgb, cx, mascara=None, tamanho=(ROCOLE_LARGURA, ROCOLE_ALTURA)):
    """Recorta a caixa cx, apaga o fundo se houver mascara e encaixa em tamanho (L, A) mantendo a proporcao
    (LANCZOS, como o cache da foto inteira), completando com FUNDO."""
    img = img_rgb.copy()
    if mascara is not None:
        img[~mascara] = FUNDO
    x0, y0, x1, y1 = cx
    pedaco = Image.fromarray(img[y0:y1, x0:x1])
    L, A = tamanho
    escala = min(L / pedaco.width, A / pedaco.height)
    nl, na = max(1, round(pedaco.width * escala)), max(1, round(pedaco.height * escala))
    tela = Image.new("RGB", (L, A), FUNDO)
    tela.paste(pedaco.resize((nl, na), Image.LANCZOS), ((L - nl) // 2, (A - na) // 2))
    return np.asarray(tela)


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


def figura_amostra(z, meta, tab, caminho, n=30):
    """Mascaras de n fotos de treino sorteadas (para conferir a olho; nada e escolhido por esta figura)."""
    amostra = meta[meta["split"] == "train"].sample(n, random_state=SEED)
    info = tab.set_index("id")
    tiles = []
    for r in amostra.itertuples():
        img = cv2.imdecode(np.frombuffer(z.read(r.arquivo), np.uint8), cv2.IMREAD_COLOR)
        arq_m = ROCOLE_MASCARAS / f"{r.id}.png"
        if arq_m.exists():
            m = cv2.resize(cv2.imread(str(arq_m), cv2.IMREAD_GRAYSCALE), (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
            cont, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cont, -1, (255, 0, 255), 8)
        t = cv2.resize(img, (384, 216), interpolation=cv2.INTER_AREA)
        i = info.loc[r.id]
        txt = f"{r.id} {'ok' if i.ok else 'FOTO INTEIRA: ' + i.motivo}"
        cv2.putText(t, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    linhas = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
    cv2.imwrite(str(caminho), np.vstack(linhas), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", default=str(ROCOLE_ZIP))
    args = ap.parse_args(argv)
    meta_csv = OUT_ROCOLE / "metadados.csv"
    if not meta_csv.exists():
        raise SystemExit(f"{meta_csv} nao existe: rode preparar_rocole.py")
    try:
        z = zipfile.ZipFile(args.zip)
    except (FileNotFoundError, zipfile.BadZipFile):
        raise SystemExit(f"zip nao encontrado ou invalido: {args.zip}. Passe --zip CAMINHO")
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if disp.type == "cpu":
        print("aviso: CUDA indisponivel, o SAM vai rodar na CPU (lento)", flush=True)
    for p in (ROCOLE_CACHE_RECORTE, ROCOLE_CACHE_RECORTE_MASCARA, ROCOLE_MASCARAS):
        p.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(meta_csv)
    sam = SegmentadorSAM(disp)
    linhas = []
    for n, r in enumerate(meta.itertuples(), 1):
        img = np.asarray(Image.open(io.BytesIO(z.read(r.arquivo))).convert("RGB"))
        h, w = img.shape[:2]
        mascaras, confiancas = sam(img)
        m, info = escolher_mascara(mascaras, confiancas)
        if m is not None:
            cx = caixa(m)
            Image.fromarray(m.astype(np.uint8) * 255).resize((w // 2, h // 2), Image.NEAREST).save(
                ROCOLE_MASCARAS / f"{r.id}.png")
            rec, rec_m = recortar(img, cx), recortar(img, cx, mascara=m)
        else:  # mascara duvidosa: entra a foto inteira nas duas entradas
            cx = (0, 0, w, h)
            rec = rec_m = recortar(img, cx)
        Image.fromarray(rec).save(ROCOLE_CACHE_RECORTE / f"{r.id}.jpg", quality=95)
        Image.fromarray(rec_m).save(ROCOLE_CACHE_RECORTE_MASCARA / f"{r.id}.jpg", quality=95)
        linhas.append({"id": r.id, "split": r.split, "classe": r.classe, **info,
                       "x0": cx[0], "y0": cx[1], "x1": cx[2], "y1": cx[3]})
        if n % 200 == 0:
            print(f"{n}/{len(meta)} fotos", flush=True)
    tab = pd.DataFrame(linhas)
    tab.to_csv(OUT_ROCOLE / "folhas_sam.csv", index=False)
    print(f"\nmascaras aceitas: {int(tab['ok'].sum())} de {len(tab)} ({tab['ok'].mean():.1%})")
    print("motivos da foto inteira:", tab.loc[~tab["ok"], "motivo"].value_counts().to_dict())
    print("aceitas por classe e split:")
    print(tab.pivot_table(index="classe", columns="split", values="ok", aggfunc="mean").round(3).to_string())
    print(f"area da folha (aceitas): mediana {tab.loc[tab['ok'], 'area'].median():.3f}")
    figura_amostra(z, meta, tab, OUT_ROCOLE / "segmentacao_amostra.jpg")
    print(f"gravado: {OUT_ROCOLE / 'folhas_sam.csv'}, {OUT_ROCOLE / 'segmentacao_amostra.jpg'}, caches e mascaras")


if __name__ == "__main__":
    main()
