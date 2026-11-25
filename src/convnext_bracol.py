"""ConvNeXt-Tiny do BRACOL: a probabilidade de cada doenca, vista na foto inteira da folha. Entra no passo 2 do
aplicativo (quais doencas), em media com as arvores de hough_doencas.py (peso 0,75 na rede).

A receita e a do primeiro modelo do BRACOL (docs/RELATORIO_BRACOL.md): ConvNeXt-Tiny pre-treinada no ImageNet,
entrada 512x256, 4 saidas multirrotulo (bicho-mineiro, ferrugem, mancha de phoma, cercosporiose) e a severidade em
5 niveis como tarefa auxiliar; AdamW, ciclo unico de LR, ate 25 epocas e parada antecipada (paciencia 8) pelo mAP de
uma validacao interna: 1/7 dos grupos das folhas de treino.

Uso (na raiz do repositorio, com PYTHONPATH=src; GPU recomendada):
  python -P src/convnext_bracol.py cv     # uma rede por dobra (as 3 dobras de hough_treinar.dobras): probabilidades
                                          # fora da dobra -> outputs/hough_bracol/convnext_oof.csv (retoma de onde parou)
  python -P src/convnext_bracol.py final  # treino + validacao -> models/bracol_convnext.pt

O teste nao e lido aqui: so por hough_doencas.py teste.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from comum import BRACOL_ALTURA, BRACOL_CACHE, BRACOL_LARGURA, DESVIO, MEDIA, MODELOS, SEED
from hough_dados import DOENCAS_BRACOL, carregar_conjunto

DOENCAS = list(DOENCAS_BRACOL)  # miner, rust, phoma, cercospora
N_SEV = 5
ARTEFATO = MODELOS / "bracol_convnext.pt"
PASTA_DOBRAS = MODELOS / "cv"  # redes das dobras: so servem para as probabilidades fora da dobra
EPOCAS, PACIENCIA, BATCH, LR = 25, 8, 16, 2e-4


# ---------------------------------------------------------------- entrada
def entrada(img_bgr):
    """Foto no formato que a rede viu no treino: deitada, 2:1 (corta o centro), 512x256, RGB (PIL). As fotos do
    BRACOL (2048x1024) so sao reduzidas."""
    if img_bgr.shape[0] > img_bgr.shape[1]:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    h, w = img_bgr.shape[:2]
    alvo = BRACOL_LARGURA / BRACOL_ALTURA
    if w / h > alvo:
        nw = max(1, round(h * alvo))
        img_bgr = img_bgr[:, (w - nw) // 2:(w - nw) // 2 + nw]
    else:
        nh = max(1, round(w / alvo))
        img_bgr = img_bgr[(h - nh) // 2:(h - nh) // 2 + nh]
    rgb = Image.fromarray(cv2.cvtColor(np.ascontiguousarray(img_bgr), cv2.COLOR_BGR2RGB))
    return rgb.resize((BRACOL_LARGURA, BRACOL_ALTURA), Image.LANCZOS)


def preparar_cache(meta, cache=BRACOL_CACHE):
    """Grava em `cache` (<id>.jpg, 512x256) as folhas de meta (colunas id, arquivo) que ainda nao estao la."""
    cache = Path(cache)
    faltam = [r for r in meta.itertuples() if not (cache / f"{r.id}.jpg").exists()]
    if not faltam:
        return
    cache.mkdir(parents=True, exist_ok=True)
    print(f"reduzindo {len(faltam)} fotos para {cache}...", flush=True)
    for r in faltam:
        img = cv2.imread(str(r.arquivo), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"nao consegui ler a foto: {r.arquivo}")
        entrada(img).save(cache / f"{r.id}.jpg", quality=95)


def transformacao(treino):
    base = [T.ToTensor(), T.Normalize(MEDIA, DESVIO)]
    if not treino:
        return T.Compose(base)
    return T.Compose([  # luz, orientacao e pequenas oclusoes
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        *base,
        T.RandomErasing(p=0.25, scale=(0.01, 0.04), value=0),
    ])


class Folhas(Dataset):
    """Folhas de meta (colunas id, as 4 doencas e severidade), lidas do cache 512x256."""

    def __init__(self, meta, treino=False, cache=BRACOL_CACHE):
        self.meta, self.tf, self.cache = meta.reset_index(drop=True), transformacao(treino), Path(cache)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]
        img = Image.open(self.cache / f"{r['id']}.jpg").convert("RGB")
        y = torch.tensor(r[DOENCAS].to_numpy(float), dtype=torch.float32)
        return self.tf(img), y, int(r["severidade"])


# ---------------------------------------------------------------- modelo
class ModeloFolha(nn.Module):
    """ConvNeXt-Tiny com duas cabecas: as 4 doencas (multirrotulo) e a severidade (5 niveis, tarefa auxiliar)."""

    def __init__(self, pretreinado=True):
        super().__init__()
        net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretreinado else None)
        self.features, self.pool = net.features, net.avgpool
        self.norm = nn.Sequential(net.classifier[0], net.classifier[1])  # LayerNorm2d + Flatten
        self.drop = nn.Dropout(0.3)
        self.cab_doenca = nn.Linear(768, len(DOENCAS))
        self.cab_sev = nn.Linear(768, N_SEV)

    def forward(self, x):
        v = self.drop(self.norm(self.pool(self.features(x))))
        return self.cab_doenca(v), self.cab_sev(v)


def dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def carregar(caminho=ARTEFATO, disp=None):
    """Rede gravada, pronta para prever (eval). FileNotFoundError se o arquivo nao existe."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} nao existe: rode convnext_bracol.py final")
    disp = disp or dispositivo()
    m = ModeloFolha(pretreinado=False)
    m.load_state_dict(torch.load(caminho, map_location=disp, weights_only=True))
    return m.to(disp).eval()


def _probs(modelo, x, disp):
    with torch.no_grad(), torch.autocast(device_type=disp.type, enabled=disp.type == "cuda"):
        logits, _ = modelo(x.to(disp))
    return torch.sigmoid(logits.float()).cpu().numpy()


def prever(modelo, meta, disp=None, cache=BRACOL_CACHE, batch=32):
    """(n, 4): probabilidade de cada doenca para as folhas de meta, na ordem de meta."""
    disp = disp or next(modelo.parameters()).device
    modelo.eval()
    saida = [_probs(modelo, x, disp) for x, _, _ in DataLoader(Folhas(meta, False, cache), batch_size=batch)]
    return np.concatenate(saida) if saida else np.zeros((0, len(DOENCAS)))


def previsor(caminho=ARTEFATO, disp=None):
    """Funcao img_bgr -> (4,) com a probabilidade de cada doenca, para o aplicativo; None se os pesos nao existem."""
    try:
        modelo = carregar(caminho, disp)
    except FileNotFoundError:
        return None
    disp = next(modelo.parameters()).device
    tf = transformacao(False)
    return lambda img_bgr: _probs(modelo, tf(entrada(img_bgr)).unsqueeze(0), disp)[0]


# ---------------------------------------------------------------- treino
def _map(y, p):
    from sklearn.metrics import average_precision_score
    return float(np.mean([average_precision_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]))


def treinar(meta_tr, meta_va, destino, epocas=EPOCAS, paciencia=PACIENCIA, batch=BATCH, lr=LR, seed=SEED,
            cache=BRACOL_CACHE, pretreinado=True, disp=None):
    """Treina em meta_tr e grava em `destino` a epoca de maior mAP em meta_va (parada antecipada). Devolve o
    historico (uma linha por epoca)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    disp = disp or dispositivo()
    dl_tr = DataLoader(Folhas(meta_tr, True, cache), batch_size=batch, shuffle=True, drop_last=len(meta_tr) > batch,
                       num_workers=2, persistent_workers=True)
    pos = meta_tr[DOENCAS].sum().to_numpy(float)
    pos_weight = torch.tensor(np.clip((len(meta_tr) - pos) / np.maximum(pos, 1), 1, 5), dtype=torch.float32).to(disp)
    cont = np.bincount(meta_tr["severidade"].astype(int), minlength=N_SEV).astype(float)
    peso_sev = torch.tensor(np.clip(cont.sum() / (N_SEV * np.maximum(cont, 1)), 0.5, 4), dtype=torch.float32).to(disp)
    modelo = ModeloFolha(pretreinado).to(disp)
    perda_d, perda_s = nn.BCEWithLogitsLoss(pos_weight=pos_weight), nn.CrossEntropyLoss(weight=peso_sev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epocas * len(dl_tr), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=disp.type == "cuda")
    y_va = meta_va[DOENCAS].to_numpy(int)
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    melhor, sem_melhora, hist = -1.0, 0, []
    for ep in range(1, epocas + 1):
        modelo.train()
        t0, soma = time.time(), 0.0
        for x, y, s in dl_tr:
            x, y, s = x.to(disp), y.to(disp), s.to(disp)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(disp.type, enabled=disp.type == "cuda"):
                a, b = modelo(x)
                perda = perda_d(a.float(), y) + 0.5 * perda_s(b.float(), s)
            scaler.scale(perda).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            soma += perda.item()
        m_ap = _map(y_va, prever(modelo, meta_va, disp, cache))
        hist.append({"epoca": ep, "perda_treino": soma / len(dl_tr), "map_validacao_interna": m_ap})
        print(f"  ep {ep:02d} perda {soma / len(dl_tr):.3f} | mAP {m_ap:.3f} | {time.time() - t0:.0f}s", flush=True)
        if m_ap > melhor:
            melhor, sem_melhora = m_ap, 0
            torch.save(modelo.state_dict(), destino)
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                print("  parada antecipada", flush=True)
                break
    return pd.DataFrame(hist)


def validacao_interna(base, seed=SEED, n=7):
    """Separa ~1/n dos grupos de `base` (estratificado pela combinacao das doencas) para a parada antecipada.
    Devolve (treino, validacao interna)."""
    from sklearn.model_selection import StratifiedGroupKFold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, va = next(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(
            base, base["estrato"], base["grupo"]))
    fora = base.index[va]
    return base.drop(index=fora), base.loc[fora]


# ---------------------------------------------------------------- comandos
def rodar_cv(conj, treinar_fn=treinar, prever_fn=None):
    """Uma rede por dobra, treinada sem as folhas da dobra; cada folha de treino + validacao e prevista so pela rede
    que nao a viu. Grava convnext_oof.csv (id, dobra, p_<doenca>) e convnext_historico.csv na pasta do conjunto."""
    import hough_treinar as ht
    meta = conj.meta
    preparar_cache(meta[meta["split"].isin(["train", "val"])])
    d = ht.dobras(meta)
    base = meta[meta["split"].isin(["train", "val"])].set_index("id").loc[d.index].rename_axis("id").reset_index()
    partes, hists = [], []
    for k in range(ht.N_DOBRAS):
        fora = base[(d == k).to_numpy()]
        tr, va = validacao_interna(base[(d != k).to_numpy()])
        destino = PASTA_DOBRAS / f"bracol_convnext_d{k}.pt"
        if destino.exists():
            print(f"dobra {k}: {destino} ja existe (retomando)", flush=True)
        else:
            print(f"dobra {k}: treino {len(tr)}, validacao interna {len(va)}, fora {len(fora)}", flush=True)
            hists.append(treinar_fn(tr, va, destino).assign(dobra=k))
        p = (prever_fn or (lambda arq, m: prever(carregar(arq), m)))(destino, fora)
        partes.append(pd.DataFrame({"id": fora["id"].to_numpy(), "dobra": k,
                                    **{f"p_{dz}": p[:, j] for j, dz in enumerate(DOENCAS)}}))
    oof = pd.concat(partes, ignore_index=True)
    conj.pasta.mkdir(parents=True, exist_ok=True)
    oof.to_csv(conj.pasta / "convnext_oof.csv", index=False)
    if hists:
        pd.concat(hists, ignore_index=True).to_csv(conj.pasta / "convnext_historico_dobras.csv", index=False)
    y = base.set_index("id").loc[oof["id"], DOENCAS].to_numpy(int)
    print(f"ConvNeXt-Tiny fora da dobra ({len(oof)} folhas): mAP {_map(y, oof[[f'p_{dz}' for dz in DOENCAS]].to_numpy()):.3f}"
          f" | gravado: {conj.pasta / 'convnext_oof.csv'}")
    return oof


def ajustar_final(conj, artefato=ARTEFATO, treinar_fn=treinar):
    """A rede do aplicativo: treino + validacao (menos 1/7 dos grupos, para a parada antecipada). Grava os pesos em
    `artefato` e, ao lado, um .json com a impressao digital da divisao e a melhor epoca."""
    meta = conj.meta
    tv = meta[meta["split"].isin(["train", "val"])].reset_index(drop=True)
    preparar_cache(tv)
    tr, va = validacao_interna(tv)
    print(f"ConvNeXt-Tiny final: treino {len(tr)}, validacao interna {len(va)}", flush=True)
    hist = treinar_fn(tr, va, artefato)
    hist.to_csv(conj.pasta / "convnext_historico_final.csv", index=False)
    melhor = hist.loc[hist["map_validacao_interna"].idxmax()]
    info = {"metadados_sha256": conj.sha256(), "melhor_epoca": int(melhor["epoca"]),
            "map_validacao_interna": float(melhor["map_validacao_interna"]), "entrada": [BRACOL_LARGURA, BRACOL_ALTURA],
            "doencas": DOENCAS}
    Path(artefato).with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"gravado: {artefato} (melhor epoca {info['melhor_epoca']}, mAP interno {info['map_validacao_interna']:.3f})")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("etapa", choices=["cv", "final"])
    args = ap.parse_args(argv)
    conj = carregar_conjunto("bracol")
    {"cv": rodar_cv, "final": ajustar_final}[args.etapa](conj)


if __name__ == "__main__":
    main()
