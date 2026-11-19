"""Treina ResNet-50 (padrao) ou ConvNeXt-Tiny no RoCoLe e grava models/rocole_<modelo>.pt.

Uso: python -P src/rocole_treinar.py [--modelo resnet50] [--epocas 25] [--epocas-cabeca 3] [--batch 16]

Laco adaptado do main.py (SIN 393): perda com pesos por classe, fine-tuning em 2 etapas (so a cabeca,
depois a rede inteira), AMP na GPU, ReduceLROnPlateau e parada antecipada pela perda de validacao.
O checkpoint gravado e o da menor perda de validacao.
"""
import argparse
import json
import platform
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from rocole_comum import CLASSES_ROCOLE, MODELOS_ROCOLE, OUT_ROCOLE, ROCOLE_CACHE, SEED
from rocole_modelo import (ENTRADAS, MODELOS_SUPORTADOS, FolhasRocole, caminho_pesos, congelar_backbone,
                           construir_modelo, nome_execucao, registrar_divisao)

LR_CABECA = 1e-3  # etapa 1: so a cabeca
LR_FINO = 1e-4  # etapa 2: rede inteira
WEIGHT_DECAY = 1e-4
PACIENCIA = 5
NUM_WORKERS = 4


def fixar_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def escolher_dispositivo():
    if torch.cuda.is_available():
        print(f"CUDA disponivel: {torch.cuda.get_device_name(0)}", flush=True)
        torch.backends.cudnn.benchmark = True  # entrada de tamanho fixo
        return torch.device("cuda")
    print("aviso: CUDA indisponivel, usando CPU (sera lento)", flush=True)
    return torch.device("cpu")


def pesos_classes(rotulos, n_classes):
    """1/contagem de cada classe, escalado para somar n_classes (como USE_CLASS_WEIGHTS do main.py)."""
    w = 1.0 / np.bincount(rotulos, minlength=n_classes)
    return w / w.sum() * n_classes


def fazer_loader(ds, treino, batch, disp):
    return DataLoader(ds, batch_size=batch, shuffle=treino, drop_last=treino, num_workers=NUM_WORKERS,
                      pin_memory=disp.type == "cuda", persistent_workers=NUM_WORKERS > 0)


def treinar_epoca(modelo, dl, criterio, otimizador, disp, scaler):
    modelo.train()
    soma, acertos, n = 0.0, 0, 0
    for x, y, _ in dl:
        x = x.to(disp, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(disp, non_blocking=True)
        otimizador.zero_grad(set_to_none=True)
        with torch.autocast(device_type=disp.type, enabled=scaler.is_enabled()):
            saida = modelo(x)
            perda = criterio(saida, y)
        scaler.scale(perda).backward()
        scaler.step(otimizador)
        scaler.update()
        soma += perda.item() * x.size(0)
        acertos += (saida.argmax(1) == y).sum().item()
        n += x.size(0)
    return soma / n, acertos / n


@torch.no_grad()
def avaliar_loader(modelo, dl, criterio, disp):
    """Perda, acuracia e F1 macro num loader de validacao."""
    modelo.eval()
    soma, n, ys, preds = 0.0, 0, [], []
    for x, y, _ in dl:
        x = x.to(disp, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(disp, non_blocking=True)
        with torch.autocast(device_type=disp.type, enabled=disp.type == "cuda"):
            saida = modelo(x)
            perda = criterio(saida, y)
        soma += perda.item() * x.size(0)
        n += x.size(0)
        ys.append(y.cpu()); preds.append(saida.argmax(1).cpu())
    ys, preds = torch.cat(ys).numpy(), torch.cat(preds).numpy()
    return soma / n, float((ys == preds).mean()), float(f1_score(ys, preds, average="macro", zero_division=0))


def criar_otimizador(modelo, lr):
    params = [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    return opt, sched


def treinar(modelo, nome, dl_tr, dl_va, criterio, disp, epocas, epocas_cabeca, destino, historico):
    """Laco de treino com 2 etapas e parada antecipada; grava o melhor checkpoint em `destino` e o historico
    em `historico` a cada epoca (uma interrupcao nao deixa checkpoint novo com historico velho)."""
    scaler = torch.amp.GradScaler("cuda", enabled=disp.type == "cuda")
    etapa = 1 if epocas_cabeca > 0 else 2
    congelar_backbone(modelo, nome, congelar=etapa == 1)
    opt, sched = criar_otimizador(modelo, LR_CABECA if etapa == 1 else LR_FINO)
    if etapa == 1:
        print(f"--- Etapa 1: so a cabeca, por {epocas_cabeca} epocas, lr={LR_CABECA} ---", flush=True)

    hist, melhor, sem_melhora = [], float("inf"), 0
    for ep in range(1, epocas + 1):
        if etapa == 1 and ep == epocas_cabeca + 1:
            etapa = 2
            congelar_backbone(modelo, nome, congelar=False)
            opt, sched = criar_otimizador(modelo, LR_FINO)
            print(f"--- Etapa 2: rede inteira, lr={LR_FINO} ---", flush=True)

        t0 = time.time()
        lr = opt.param_groups[0]["lr"]
        perda_tr, acc_tr = treinar_epoca(modelo, dl_tr, criterio, opt, disp, scaler)
        perda_va, acc_va, f1_va = avaliar_loader(modelo, dl_va, criterio, disp)
        sched.step(perda_va)
        dt = time.time() - t0
        hist.append({"epoca": ep, "etapa": etapa, "perda_treino": perda_tr, "acc_treino": acc_tr,
                     "perda_val": perda_va, "acc_val": acc_va, "f1_macro_val": f1_va, "lr": lr, "tempo_s": dt})
        pd.DataFrame(hist).to_csv(historico, index=False)
        print(f"Epoca {ep:02d}/{epocas} | treino perda {perda_tr:.4f} acc {acc_tr:.4f} | "
              f"val perda {perda_va:.4f} acc {acc_va:.4f} F1 macro {f1_va:.4f} | {dt:.0f}s", flush=True)

        if perda_va < melhor:
            melhor, sem_melhora = perda_va, 0
            torch.save(modelo.state_dict(), destino)
        else:
            sem_melhora += 1
            if sem_melhora >= PACIENCIA:
                print(f"parada antecipada na epoca {ep}", flush=True)
                break
    return hist


def plotar_historico(hist, caminho):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(hist["epoca"], hist["perda_treino"], label="Treino")
    a1.plot(hist["epoca"], hist["perda_val"], label="Validacao")
    a1.set_xlabel("Epoca"); a1.set_ylabel("Perda"); a1.set_title("Perda"); a1.legend()
    a2.plot(hist["epoca"], hist["acc_treino"], label="Treino")
    a2.plot(hist["epoca"], hist["acc_val"], label="Validacao")
    a2.plot(hist["epoca"], hist["f1_macro_val"], "--", label="Validacao (F1 macro)")
    a2.set_xlabel("Epoca"); a2.set_ylabel("Acuracia / F1"); a2.set_title("Acuracia"); a2.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", choices=MODELOS_SUPORTADOS, default="resnet50")
    ap.add_argument("--epocas", type=int, default=25)
    ap.add_argument("--epocas-cabeca", type=int, default=3, help="epocas da etapa 1 (0 desliga)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--entrada", choices=list(ENTRADAS), default="foto",
                    help="foto inteira, folha recortada ou folha recortada com fundo apagado (rocole_folha.py)")
    args = ap.parse_args(argv)

    fixar_seed(args.seed)
    cache = ENTRADAS[args.entrada]
    if not cache.exists():
        raise SystemExit(f"cache {cache} nao existe: rode {'rocole_preparar.py' if args.entrada == 'foto' else 'rocole_folha.py'}")
    nome = nome_execucao(args.modelo, args.entrada)
    disp = escolher_dispositivo()
    print("ambiente:", json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                                   "torchvision": torchvision.__version__, "cuda": torch.version.cuda}), flush=True)
    print(f"modelo: {args.modelo} | entrada: {args.entrada} | epocas: {args.epocas} | etapa 1: {args.epocas_cabeca} | batch: {args.batch}",
          flush=True)

    ds_tr, ds_va = FolhasRocole("train", treino=True, cache=cache), FolhasRocole("val", cache=cache)
    dl_tr, dl_va = fazer_loader(ds_tr, True, args.batch, disp), fazer_loader(ds_va, False, args.batch, disp)
    print(f"imagens: treino {len(ds_tr)} | validacao {len(ds_va)}", flush=True)

    rotulos = ds_tr.meta["classe"].map(CLASSES_ROCOLE.index).to_numpy()
    w = pesos_classes(rotulos, len(CLASSES_ROCOLE))
    print("pesos das classes na perda:", {c: round(float(v), 3) for c, v in zip(CLASSES_ROCOLE, w)}, flush=True)
    criterio = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=disp))

    modelo = construir_modelo(args.modelo, len(CLASSES_ROCOLE)).to(disp).to(memory_format=torch.channels_last)
    MODELOS_ROCOLE.mkdir(exist_ok=True)
    registrar_divisao(caminho_pesos(nome))  # liga o checkpoint a divisao usada (ver verificar_divisao)
    pasta = OUT_ROCOLE / nome
    pasta.mkdir(parents=True, exist_ok=True)

    hist = pd.DataFrame(treinar(modelo, args.modelo, dl_tr, dl_va, criterio, disp, args.epocas,
                                args.epocas_cabeca, caminho_pesos(nome), pasta / "historico_treino.csv"))
    plotar_historico(hist, pasta / "historico_treino.png")
    m = hist.loc[hist["perda_val"].idxmin()]
    print(f"melhor epoca: {int(m['epoca'])} | val perda {m['perda_val']:.4f} acc {m['acc_val']:.4f} "
          f"F1 macro {m['f1_macro_val']:.4f} | pesos: {caminho_pesos(nome)}", flush=True)


if __name__ == "__main__":
    main()
