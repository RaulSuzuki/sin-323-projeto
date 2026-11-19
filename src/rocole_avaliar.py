"""Avalia o modelo do RoCoLe: metricas no teste, IC por bootstrap de plantas, matriz de confusao e Grad-CAM.

Uso: python -P src/rocole_avaliar.py [--modelo resnet50] [--pesos models/rocole_resnet50.pt]

O teste e pontuado uma vez por modelo treinado. A validacao entra so na tabela de comparacao
(outputs/comparacao_modelos.csv), que e onde se escolhe entre modelos.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from rocole_comum import CLASSES_ROCOLE, NOMES_ROCOLE_PT, OUT_ROCOLE, ROCOLE_CACHE, SEED
from rocole_gradcam import gradcam_camada
from rocole_modelo import (ENTRADAS, MODELOS_SUPORTADOS, FolhasRocole, camada_alvo, carregar, nome_execucao,
                           transformacao)

K = len(CLASSES_ROCOLE)
ROTULOS_PT = [NOMES_ROCOLE_PT[c] for c in CLASSES_ROCOLE]
CHAVES_IC = ["acuracia", "f1_macro"] + [f"f1_{c}" for c in CLASSES_ROCOLE]


@torch.no_grad()
def prever(modelo, dl, disp):
    """Probabilidades (N, 3), classes reais (N,) e ids."""
    modelo.eval()
    probs, ys, ids = [], [], []
    for x, y, i in dl:
        with torch.autocast(device_type=disp.type, enabled=disp.type == "cuda"):
            saida = modelo(x.to(disp))
        probs.append(torch.softmax(saida.float(), 1).cpu())
        ys.append(y)
        ids += list(i)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy(), ids


def metricas(y, pred):
    """Acuracia, precisao/recall/F1 macro e weighted e F1 por classe."""
    out = {"acuracia": float((y == pred).mean())}
    for media in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(K)), average=media, zero_division=0)
        out.update({f"precisao_{media}": float(p), f"recall_{media}": float(r), f"f1_{media}": float(f)})
    f1s = precision_recall_fscore_support(y, pred, labels=list(range(K)), average=None, zero_division=0)[2]
    out.update({f"f1_{c}": float(v) for c, v in zip(CLASSES_ROCOLE, f1s)})
    return out


def _acc_f1(y, pred):
    """Acuracia, F1 macro e F1 por classe pela matriz de confusao (rapido, para o bootstrap)."""
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)  # = 2tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros(K), where=den > 0)
    return [tp.sum() / cm.sum(), f1.mean(), *f1]


def bootstrap_por_grupo(y, pred, grupos, b=2000, seed=SEED):
    """IC 95% reamostrando grupos (plantas) com reposicao, com todas as imagens de cada grupo sorteado.
    Os dois lados de uma folha nao sao independentes; reamostrar imagens daria IC estreito demais."""
    rng = np.random.default_rng(seed)
    unicos = np.unique(grupos)
    por_grupo = [np.flatnonzero(grupos == g) for g in unicos]
    amostras = []
    for _ in range(b):
        i = np.concatenate([por_grupo[k] for k in rng.integers(0, len(unicos), len(unicos))])
        amostras.append(_acc_f1(y[i], pred[i]))
    boot = pd.DataFrame(amostras, columns=CHAVES_IC)
    return pd.DataFrame({"estimativa": pd.Series(_acc_f1(y, pred), index=CHAVES_IC),
                         "IC95_inf": boot.quantile(0.025), "IC95_sup": boot.quantile(0.975)})


def _p_predita(df):
    p = df[[f"p_{c}" for c in CLASSES_ROCOLE]].to_numpy()
    return p[np.arange(len(df)), df["classe_predita"].map(CLASSES_ROCOLE.index).to_numpy()]


def escolher_acertos(pred_df, n=3, seed=SEED):
    """Ate n acertos por classe, sorteados (nao os mais confiantes), na ordem de CLASSES_ROCOLE."""
    partes = []
    for c in CLASSES_ROCOLE:
        ok = pred_df[pred_df["acertou"] & (pred_df["classe_real"] == c)]
        partes.append(ok.sample(n=min(n, len(ok)), random_state=seed))
    return pd.concat(partes)


def escolher_erros(pred_df, n=8):
    """Os n erros com maior probabilidade na classe predita (erros mais confiantes primeiro)."""
    erros = pred_df[~pred_df["acertou"]].copy()
    erros["p_predita"] = _p_predita(erros)
    return erros.sort_values("p_predita", ascending=False, kind="stable").head(n)


def epoca_escolhida(hist_csv, pesos_customizados):
    """Epoca do primeiro minimo da perda de validacao. None com --pesos (o historico pode ser de outro treino)."""
    if pesos_customizados or not hist_csv.exists():
        return None
    hist = pd.read_csv(hist_csv)
    return int(hist.loc[hist["perda_val"].idxmin(), "epoca"])


def grupos_por_id(ids, meta):
    """Grupo (planta) de cada predicao, casado pelo id, sem depender da ordem do DataLoader."""
    g = meta.set_index("id")["grupo"]
    assert pd.Index(ids).isin(g.index).all(), "predicoes com id fora do metadados"
    return g.loc[list(ids)].to_numpy()


def atualizar_comparacao(caminho, linha):
    """Grava/atualiza a linha do modelo em comparacao_modelos.csv, mantendo a ordem das demais."""
    nova = pd.DataFrame([linha])
    if caminho.exists():
        tab = pd.read_csv(caminho)
        pos = np.flatnonzero(tab["modelo"] == linha["modelo"])
        if len(pos):
            tab = pd.concat([tab.iloc[:pos[0]], nova, tab.iloc[pos[0] + 1:]], ignore_index=True)
        else:
            tab = pd.concat([tab, nova], ignore_index=True)
    else:
        tab = nova
    tab.to_csv(caminho, index=False)
    return tab


def plotar_matriz_confusao(y, pred, caminho, nome):
    cm = confusion_matrix(y, pred, labels=list(range(K)))
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(K), ROTULOS_PT, rotation=20)
    ax.set_yticks(range(K), ROTULOS_PT)
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusao (teste) - {nome}")
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def _mapa(modelo, camada, id_, classe, disp, cache=ROCOLE_CACHE):
    """Imagem do cache (para mostrar) e Grad-CAM da classe pedida."""
    with Image.open(cache / f"{id_}.jpg") as im:
        img = im.convert("RGB")
    x = transformacao(False)(img).unsqueeze(0).to(disp)
    cam, _ = gradcam_camada(modelo, camada, x, classe)
    return np.asarray(img), cam


def _painel(ax, img, cam=None, titulo="", cor="black"):
    ax.imshow(img)
    if cam is not None:
        ax.imshow(cam, cmap="jet", alpha=0.45)
    ax.set_title(titulo, fontsize=8, color=cor)
    ax.axis("off")


def figura_acertos(modelo, nome, sel, disp, caminho, n=3, cache=ROCOLE_CACHE):
    camada = camada_alvo(modelo, nome)
    fig, axes = plt.subplots(K, 2 * n, figsize=(2 * n * 2.6, K * 1.9), layout="constrained")
    for ax in axes.ravel():
        ax.axis("off")
    for lin, c in enumerate(CLASSES_ROCOLE):
        for col, r in enumerate(sel[sel["classe_real"] == c].itertuples()):
            img, cam = _mapa(modelo, camada, r.id, CLASSES_ROCOLE.index(c), disp, cache)
            p = getattr(r, f"p_{c}")
            _painel(axes[lin, 2 * col], img, titulo=f"{r.id} | Real: {NOMES_ROCOLE_PT[c]}")
            _painel(axes[lin, 2 * col + 1], img, cam, f"Grad-CAM: {NOMES_ROCOLE_PT[c]} (p={p:.2f})", "green")
    fig.suptitle(f"Grad-CAM em acertos do teste - {nome} (sorteados, {n} por classe)")
    fig.savefig(caminho, dpi=110)
    plt.close(fig)


def figura_erros(modelo, nome, erros, disp, caminho, cache=ROCOLE_CACHE):
    camada = camada_alvo(modelo, nome)
    if erros.empty:
        fig, ax = plt.subplots(figsize=(4, 1.5), layout="constrained")
        ax.text(0.5, 0.5, "nenhum erro no teste", ha="center", va="center")
        ax.axis("off")
    else:
        fig, axes = plt.subplots(len(erros), 3, figsize=(3 * 3.2, len(erros) * 2.0), squeeze=False,
                                 layout="constrained")
        for lin, r in enumerate(erros.itertuples()):
            real, pred = r.classe_real, r.classe_predita
            img, cam_pred = _mapa(modelo, camada, r.id, CLASSES_ROCOLE.index(pred), disp, cache)
            _, cam_real = _mapa(modelo, camada, r.id, CLASSES_ROCOLE.index(real), disp, cache)
            _painel(axes[lin, 0], img, titulo=f"{r.id} | Real: {NOMES_ROCOLE_PT[real]}")
            _painel(axes[lin, 1], img, cam_pred, f"Predito: {NOMES_ROCOLE_PT[pred]} (p={r.p_predita:.2f})", "red")
            _painel(axes[lin, 2], img, cam_real,
                    f"Mapa da classe real: {NOMES_ROCOLE_PT[real]} (p={getattr(r, f'p_{real}'):.2f})")
    fig.suptitle(f"Erros mais confiantes do teste - {nome}")
    fig.savefig(caminho, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", choices=MODELOS_SUPORTADOS, default="resnet50")
    ap.add_argument("--pesos", type=Path, default=None, help="checkpoint (padrao: models/rocole_<modelo>.pt)")
    ap.add_argument("--entrada", choices=list(ENTRADAS), default="foto")
    args = ap.parse_args(argv)
    nome = nome_execucao(args.modelo, args.entrada)
    cache = ENTRADAS[args.entrada]
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if disp.type == "cpu":
        print("aviso: CUDA indisponivel, usando CPU (sera lento)", flush=True)
    try:
        modelo = carregar(args.modelo, disp, args.pesos, args.entrada)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    pasta = OUT_ROCOLE / nome
    pasta.mkdir(parents=True, exist_ok=True)

    ds_va, ds_te = FolhasRocole("val", cache=cache), FolhasRocole("test", cache=cache)
    pv, yv, _ = prever(modelo, DataLoader(ds_va, batch_size=32), disp)
    pt, yt, ids = prever(modelo, DataLoader(ds_te, batch_size=32), disp)
    predv, predt = pv.argmax(1), pt.argmax(1)

    m = metricas(yt, predt)
    mv = metricas(yv, predv)
    ic = bootstrap_por_grupo(yt, predt, grupos_por_id(ids, ds_te.meta))
    relatorio = classification_report(yt, predt, labels=list(range(K)), target_names=ROTULOS_PT, digits=4,
                                      zero_division=0)
    epoca = epoca_escolhida(pasta / "historico_treino.csv", pesos_customizados=args.pesos is not None)

    pred_df = pd.DataFrame({"id": ids, "classe_real": [CLASSES_ROCOLE[k] for k in yt],
                            "classe_predita": [CLASSES_ROCOLE[k] for k in predt],
                            **{f"p_{c}": pt[:, k] for k, c in enumerate(CLASSES_ROCOLE)}})
    pred_df["acertou"] = pred_df["classe_real"] == pred_df["classe_predita"]
    pred_df.to_csv(pasta / "predicoes_teste.csv", index=False)

    print(f"\n--- Resultados no teste ({nome}, {len(yt)} imagens) ---")
    print(f"Acuracia: {m['acuracia']:.4f}")
    print(f"Precisao (macro): {m['precisao_macro']:.4f} | Precisao (weighted): {m['precisao_weighted']:.4f}")
    print(f"Recall (macro): {m['recall_macro']:.4f} | Recall (weighted): {m['recall_weighted']:.4f}")
    print(f"F1 (macro): {m['f1_macro']:.4f} | F1 (weighted): {m['f1_weighted']:.4f}")
    print("\nRelatorio por classe:\n" + relatorio)
    print(f"IC 95% (bootstrap de {ds_te.meta['grupo'].nunique()} grupos de plantas, 2000 reamostragens):")
    print(ic.round(4).to_string())
    print(f"\nValidacao ({len(yv)} imagens): acuracia {mv['acuracia']:.4f} | F1 macro {mv['f1_macro']:.4f}")

    with open(pasta / "relatorio_classificacao.txt", "w", encoding="utf-8") as f:
        f.write(f"Modelo: {nome} | teste: {len(yt)} imagens\n")
        f.write("Metrica principal: F1 macro (classes desbalanceadas; acaro-vermelho e ~10% das imagens)\n\n")
        f.write(relatorio)
    ic.to_csv(pasta / "teste_intervalos_confianca.csv")
    resultados = {"modelo": nome, "pesos": str(args.pesos or f"models/rocole_{nome}.pt"), "epoca_escolhida": epoca,
                  "n_teste": int(len(yt)), "n_val": int(len(yv)), "teste": m,
                  "val": {"acuracia": mv["acuracia"], "f1_macro": mv["f1_macro"]},
                  "matriz_confusao_teste": confusion_matrix(yt, predt, labels=list(range(K))).tolist(),
                  "classes": CLASSES_ROCOLE}
    with open(pasta / "resultados_teste.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)

    plotar_matriz_confusao(yt, predt, pasta / "matriz_confusao.png", nome)
    figura_acertos(modelo, nome, escolher_acertos(pred_df), disp, pasta / "gradcam_acertos.png", cache=cache)
    figura_erros(modelo, nome, escolher_erros(pred_df), disp, pasta / "gradcam_erros.png", cache=cache)

    atualizar_comparacao(OUT_ROCOLE / "comparacao_modelos.csv", {
        "modelo": nome, "val_acuracia": round(mv["acuracia"], 4), "val_f1_macro": round(mv["f1_macro"], 4),
        "teste_acuracia": round(m["acuracia"], 4), "teste_f1_macro": round(m["f1_macro"], 4),
        "teste_f1_weighted": round(m["f1_weighted"], 4), "epoca_escolhida": epoca})
    print(f"\nResultados salvos em: {pasta}")


if __name__ == "__main__":
    main()
