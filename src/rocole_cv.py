"""Validacao cruzada agrupada (3 dobras por planta) no treino + validacao, para comparar entradas da CNN:
foto inteira, folha-alvo recortada e folha recortada com fundo apagado. O teste nao entra.

Uso: python -P src/rocole_cv.py [--modelo resnet50]   (retoma de onde parou: execucoes ja feitas sao puladas)

Em cada dobra, 1/3 das plantas fica de fora para medir; do resto, ~1/7 das plantas vira a validacao interna da
parada antecipada. A mesma receita de rocole_treinar.py, a mesma seed por dobra para as 3 entradas.

Regra fixada antes de rodar: uma entrada recortada so substitui a foto inteira se o F1 macro medio (fora da dobra)
for maior E se ela vencer a foto inteira em pelo menos 2 das 3 dobras. Se as duas passarem, fica a de maior media.
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from rocole_comum import CLASSES_ROCOLE, MODELOS_ROCOLE, OUT_ROCOLE, SEED
from rocole_avaliar import metricas, prever
from rocole_modelo import ENTRADAS, MODELOS_SUPORTADOS, FolhasRocole, construir_modelo
from rocole_treinar import escolher_dispositivo, fazer_loader, fixar_seed, pesos_classes, treinar
from rocole_validacao import avaliar_folhas, recall_por_classe

N_DOBRAS = 3
PASTA = OUT_ROCOLE / "cv_entradas"


def dobras_cv(meta, n_dobras=N_DOBRAS, seed=SEED):
    """Para cada dobra, uma copia do treino + validacao com split train/val/test (test = plantas de fora)."""
    base = meta[meta["split"].isin(["train", "val"])].reset_index(drop=True)
    saida = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        externas = StratifiedGroupKFold(n_splits=n_dobras, shuffle=True, random_state=seed)
        for resto_idx, fora_idx in externas.split(base, base["classe"], base["grupo"]):
            d = base.copy()
            d.loc[fora_idx, "split"] = "test"
            resto = d.iloc[resto_idx]
            internas = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
            tr, va = next(internas.split(resto, resto["classe"], resto["grupo"]))
            d.loc[resto.index[tr], "split"] = "train"
            d.loc[resto.index[va], "split"] = "val"
            saida.append(d)
    return saida


def decidir(res):
    """res: entrada, dobra, f1_macro. Aplica a regra fixada no cabecalho; devolve a entrada escolhida e o porque."""
    tab = res.pivot_table(index="dobra", columns="entrada", values="f1_macro")
    medias = tab.mean()
    candidatas = {}
    for e in [c for c in tab.columns if c != "foto"]:
        vitorias = int((tab[e] > tab["foto"]).sum())
        if medias[e] > medias["foto"] and vitorias >= 2:
            candidatas[e] = medias[e]
    escolhida = max(candidatas, key=candidatas.get) if candidatas else "foto"
    return {"escolhida": escolhida, "medias": {k: float(v) for k, v in medias.items()},
            "vitorias_sobre_foto": {e: int((tab[e] > tab["foto"]).sum()) for e in tab.columns if e != "foto"},
            "aprovadas": sorted(candidatas)}


def rodar(modelo_nome, entrada, k, meta_k, disp, batch=16):
    """Treina numa dobra e mede nas plantas de fora. Devolve uma linha de resultados e as predicoes."""
    fixar_seed(SEED + k)  # mesma seed por dobra para as 3 entradas
    cache = ENTRADAS[entrada]
    ds_tr = FolhasRocole("train", treino=True, meta=meta_k, cache=cache)
    ds_va = FolhasRocole("val", meta=meta_k, cache=cache)
    ds_fora = FolhasRocole("test", meta=meta_k, cache=cache)
    rot = ds_tr.meta["classe"].map(CLASSES_ROCOLE.index).to_numpy()
    w = torch.tensor(pesos_classes(rot, len(CLASSES_ROCOLE)), dtype=torch.float32, device=disp)
    modelo = construir_modelo(modelo_nome, len(CLASSES_ROCOLE)).to(disp).to(memory_format=torch.channels_last)
    pesos = MODELOS_ROCOLE / "cv" / f"{modelo_nome}_{entrada}_d{k}.pt"
    pesos.parent.mkdir(parents=True, exist_ok=True)
    treinar(modelo, modelo_nome, fazer_loader(ds_tr, True, batch, disp), fazer_loader(ds_va, False, batch, disp),
            nn.CrossEntropyLoss(weight=w), disp, 25, 3, pesos, PASTA / f"historico_{entrada}_d{k}.csv")
    modelo.load_state_dict(torch.load(pesos, map_location=disp, weights_only=True))
    probs, y, ids = prever(modelo, DataLoader(ds_fora, batch_size=32), disp)
    pred = probs.argmax(1)
    m = metricas(y, pred)
    pred_df = pd.DataFrame({"id": ids, "classe": [CLASSES_ROCOLE[i] for i in y],
                            "classe_predita": [CLASSES_ROCOLE[i] for i in pred], "entrada": entrada, "dobra": k})
    folhas = avaliar_folhas(pred_df).set_index(["estrategia", "doenca"])
    linha = {"entrada": entrada, "dobra": k, "n_fora": len(y), "f1_macro": m["f1_macro"], "acuracia": m["acuracia"],
             **{f"f1_{c}": m[f"f1_{c}"] for c in CLASSES_ROCOLE},
             **{f"recall_{c}": v for c, v in recall_por_classe(y, pred).items()},
             "folha_duas_fotos_acaro_sens": folhas.loc[("duas_fotos", "red_spider_mite"), "sensibilidade"],
             "folha_duas_fotos_ferrugem_sens": folhas.loc[("duas_fotos", "rust"), "sensibilidade"]}
    return linha, pred_df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", choices=MODELOS_SUPORTADOS, default="resnet50")
    args = ap.parse_args(argv)
    for e, c in ENTRADAS.items():
        if not c.exists():
            raise SystemExit(f"cache da entrada '{e}' nao existe ({c}): rode rocole_preparar.py e rocole_folha.py")
    PASTA.mkdir(parents=True, exist_ok=True)
    disp = escolher_dispositivo()
    meta = pd.read_csv(OUT_ROCOLE / "metadados.csv")
    dobras = dobras_cv(meta)
    arq_res, arq_pred = PASTA / "resultados.csv", PASTA / "predicoes_fora_da_dobra.csv"
    res = pd.read_csv(arq_res) if arq_res.exists() else pd.DataFrame(columns=["entrada", "dobra"])
    for k, meta_k in enumerate(dobras):
        for entrada in ENTRADAS:
            if ((res["entrada"] == entrada) & (res["dobra"] == k)).any():
                print(f"ja feito: {entrada}, dobra {k}", flush=True)
                continue
            print(f"\n=== entrada '{entrada}', dobra {k + 1}/{N_DOBRAS} ===", flush=True)
            linha, pred_df = rodar(args.modelo, entrada, k, meta_k, disp)
            res = pd.concat([res, pd.DataFrame([linha])], ignore_index=True)
            res.to_csv(arq_res, index=False)
            pred_df.to_csv(arq_pred, mode="a", header=not arq_pred.exists(), index=False)
            print(f"F1 macro fora da dobra: {linha['f1_macro']:.4f} | recall acaro {linha['recall_red_spider_mite']:.3f}",
                  flush=True)

    dec = decidir(res)
    with open(PASTA / "decisao.json", "w", encoding="utf-8") as f:
        json.dump(dec, f, indent=2)
    colunas = ["f1_macro", "acuracia", "recall_red_spider_mite", "recall_rust", "folha_duas_fotos_acaro_sens"]
    print("\n--- Validacao cruzada (3 dobras por planta, treino + validacao) ---")
    print(res.pivot_table(index="dobra", columns="entrada", values="f1_macro").round(4).to_string())
    print("\nmedia (desvio) por entrada:")
    print(res.groupby("entrada")[colunas].agg(["mean", "std"]).round(4).to_string())
    print(f"\nregra: vitorias sobre a foto inteira {dec['vitorias_sobre_foto']} | aprovadas: {dec['aprovadas']}")
    print(f"entrada escolhida: {dec['escolhida']}")


if __name__ == "__main__":
    main()
