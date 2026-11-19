"""Ajustes avaliados so na validacao, sem tocar no teste: deslocamentos por classe e diagnostico pela folha inteira.

Uso: python -P src/rocole_validacao.py [--modelo resnet50]

1. Deslocamentos por classe: soma b_c ao log da probabilidade de cada classe (b_saudavel = 0) e escolhe b_acaro e
   b_ferrugem pelo F1 macro. A validacao tem so 22 fotos de acaro: escolher e medir nas mesmas fotos superestimaria o
   ganho. Por isso o ganho e estimado com validacao cruzada agrupada por planta dentro da validacao (escolhe em 4/5,
   mede no 1/5 restante). Os deslocamentos escolhidos na validacao inteira ficam gravados para o modelo final.
2. Folha inteira: cada folha tem foto da face (H) e do verso (E). Compara diagnosticar com 1 foto (so a face ou so o
   verso) e com as 2 ("doenca presente se aparecer em qualquer lado").

O teste nao e usado aqui: estes ajustes so serao pontuados no teste junto com o modelo final.

Uso no teste, uma vez, com o modelo final: python -P src/rocole_validacao.py --teste
Le as predicoes gravadas por rocole_avaliar.py (nao roda o modelo) e mede a folha inteira com IC por planta.
"""
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from rocole_comum import CLASSES_ROCOLE, NOMES_ROCOLE_PT, OUT_ROCOLE, SEED
from rocole_avaliar import _acc_f1, grupos_por_id, metricas, prever
from rocole_modelo import MODELOS_SUPORTADOS, FolhasRocole, carregar
from rocole_regras import bootstrap_binario

K = len(CLASSES_ROCOLE)
GRADE = np.round(np.arange(-2.0, 2.0001, 0.25), 2)  # deslocamentos testados para acaro e ferrugem


def aplicar_deslocamentos(probs, desl):
    """Classe predita depois de somar desl[c] ao log da probabilidade de cada classe."""
    return (np.log(np.clip(probs, 1e-9, 1.0)) + np.asarray(desl)).argmax(1)


def escolher_deslocamentos(probs, y, grade=GRADE):
    """(0, b_acaro, b_ferrugem) com o maior F1 macro; empate: o menor |b_acaro| + |b_ferrugem| (mais perto do argmax)."""
    melhor, chave_melhor = None, None
    for bm in grade:
        for br in grade:
            d = np.array([0.0, bm, br])
            chave = (round(_acc_f1(y, aplicar_deslocamentos(probs, d))[1], 12), -(abs(bm) + abs(br)))
            if chave_melhor is None or chave > chave_melhor:
                melhor, chave_melhor = d, chave
    return melhor


def recall_por_classe(y, pred):
    return {c: float((pred[y == k] == k).mean()) if (y == k).any() else float("nan") for k, c in enumerate(CLASSES_ROCOLE)}


def deslocamentos_cv(probs, y, grupos, n_dobras=5, seed=SEED):
    """Ganho honesto dos deslocamentos: escolhidos nas outras dobras, aplicados na dobra de fora (agrupada por planta)."""
    pred = np.empty(len(y), int)
    por_dobra = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dobras = StratifiedGroupKFold(n_splits=n_dobras, shuffle=True, random_state=seed).split(probs, y, grupos)
        for tr, te in dobras:
            d = escolher_deslocamentos(probs[tr], y[tr])
            pred[te] = aplicar_deslocamentos(probs[te], d)
            por_dobra.append([float(v) for v in d[1:]])
    base = probs.argmax(1)
    ma, mj = metricas(y, base), metricas(y, pred)
    return {"pred_fora_da_dobra": pred, "f1_macro_argmax": ma["f1_macro"], "f1_macro_ajustado": mj["f1_macro"],
            "argmax": {**ma, "recall": recall_por_classe(y, base)},
            "ajustado": {**mj, "recall": recall_por_classe(y, pred)},
            "deslocamentos_por_dobra_acaro_ferrugem": por_dobra}


def folha_id(id_):
    """'C10P10E2' e 'C10P10H2' -> 'C10P10F2' (mesma folha, lados diferentes). Nome fora do padrao -> None."""
    m = re.fullmatch(r"(C\d+P\d+)[HE](\d+)", id_)
    return f"{m[1]}F{m[2]}" if m else None


ESTRATEGIAS = ["so_face", "so_verso", "duas_fotos"]
DOENCAS = ["rust", "red_spider_mite"]


def _folhas_completas(df):
    """Fotos das folhas que tem face (H) e verso (E), com as colunas 'folha' e 'lado'."""
    d = df.assign(folha=df["id"].map(folha_id), lado=df["id"].str.extract(r"C\d+P\d+([HE])\d+$")[0])
    d = d.dropna(subset=["folha", "lado"])
    completas = d.groupby("folha")["lado"].nunique()
    return d[d["folha"].isin(completas[completas == 2].index)]


def presenca_folhas(df):
    """df: id, classe, classe_predita. Uma linha por folha com as duas fotos e por doenca: 'verdade' (a doenca aparece
    em algum lado) e o predito por cada estrategia: so a foto da face, so a do verso, as duas (qualquer lado)."""
    d = _folhas_completas(df)
    face, verso = d[d["lado"] == "H"].set_index("folha"), d[d["lado"] == "E"].set_index("folha")
    partes = []
    for doenca in DOENCAS:
        verdade = d.groupby("folha")["classe"].apply(lambda s: (s == doenca).any())
        pred = {"so_face": face["classe_predita"].eq(doenca), "so_verso": verso["classe_predita"].eq(doenca),
                "duas_fotos": d.groupby("folha")["classe_predita"].apply(lambda s: (s == doenca).any())}
        partes.append(pd.DataFrame({"doenca": doenca, "verdade": verdade.astype(bool),
                                    **{n: p.reindex(verdade.index).astype(bool) for n, p in pred.items()}}))
    return pd.concat(partes).rename_axis("folha").reset_index()


def avaliar_folhas(df):
    """Para cada doenca e estrategia, sensibilidade e especificidade por folha. 'acerto_folha' = fracao das folhas com
    ferrugem E acaro (presenca/ausencia) certos ao mesmo tempo."""
    pres = presenca_folhas(df)
    linhas, acertos = [], {}
    for doenca in DOENCAS:
        p_d = pres[pres["doenca"] == doenca]
        v = p_d["verdade"].to_numpy()
        for nome in ESTRATEGIAS:
            p = p_d[nome].to_numpy()
            acertos[nome] = acertos.get(nome, np.ones(len(v), bool)) & (p == v)
            tp, fn, tn, fp = (v & p).sum(), (v & ~p).sum(), (~v & ~p).sum(), (~v & p).sum()
            linhas.append({"estrategia": nome, "doenca": doenca, "folhas": int(len(v)), "folhas_com_doenca": int(v.sum()),
                           "sensibilidade": tp / (tp + fn) if tp + fn else float("nan"),
                           "especificidade": tn / (tn + fp) if tn + fp else float("nan")})
    tab = pd.DataFrame(linhas)
    tab["acerto_folha"] = tab["estrategia"].map({n: float(a.mean()) for n, a in acertos.items()})
    return tab


def folhas_com_ic(df, meta, b=2000):
    """Sensibilidade e especificidade por folha, com IC 95% por bootstrap de plantas (coluna 'grupo' do metadados)."""
    pres = presenca_folhas(df)
    d = _folhas_completas(df)
    grupo_foto = d.assign(grupo=grupos_por_id(d["id"], meta)).groupby("folha")["grupo"]
    assert (grupo_foto.nunique() == 1).all(), "face e verso de uma folha em grupos diferentes"
    pres["grupo"] = pres["folha"].map(grupo_foto.first())
    linhas = []
    for doenca in DOENCAS:
        p_d = pres[pres["doenca"] == doenca]
        for nome in ESTRATEGIAS:
            tab = bootstrap_binario(p_d["verdade"], p_d[nome], p_d["grupo"], b=b)
            for m in ["sensibilidade", "especificidade"]:
                linhas.append({"estrategia": nome, "doenca": doenca, "metrica": m, **tab.loc[m].to_dict()})
    return pd.DataFrame(linhas)


def verificar_ids_teste(ids, meta):
    """As predicoes do teste tem de ser exatamente as fotos de teste da divisao atual."""
    if set(ids) != set(meta.loc[meta["split"] == "test", "id"]):
        raise SystemExit("predicoes_teste.csv nao bate com o teste de metadados.csv: rode rocole_avaliar.py de novo")


def pontuar_teste(modelo_nome):
    """Diagnostico pela folha inteira no teste, a partir das predicoes ja gravadas por rocole_avaliar.py (argmax,
    sem deslocamentos). A estrategia 'duas_fotos' foi escolhida na validacao; as outras ficam para comparar."""
    pasta = OUT_ROCOLE / modelo_nome
    arq = pasta / "predicoes_teste.csv"
    if not arq.exists():
        raise SystemExit(f"{arq} nao existe: rode rocole_avaliar.py antes")
    pred = pd.read_csv(arq).rename(columns={"classe_real": "classe"})
    meta = pd.read_csv(OUT_ROCOLE / "metadados.csv")
    verificar_ids_teste(pred["id"], meta)
    folhas = avaliar_folhas(pred)
    ic = folhas_com_ic(pred, meta)
    folhas.to_csv(pasta / "folhas_teste.csv", index=False)
    ic.to_csv(pasta / "folhas_teste_ic.csv", index=False)
    print(f"\n--- Folha inteira (TESTE, {folhas['folhas'].iloc[0]} folhas com as duas fotos) ---")
    print(folhas.round(4).to_string(index=False))
    print("\nIC 95% por bootstrap de plantas:")
    print(ic.round(4).to_string(index=False))
    print(f"\nResultados salvos em: {pasta}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modelo", choices=MODELOS_SUPORTADOS, default="resnet50")
    ap.add_argument("--teste", action="store_true",
                    help="so para o modelo final: pontua no teste o diagnostico pela folha inteira")
    args = ap.parse_args(argv)
    if args.teste:
        return pontuar_teste(args.modelo)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        modelo = carregar(args.modelo, disp)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    pasta = OUT_ROCOLE / args.modelo
    pasta.mkdir(parents=True, exist_ok=True)

    ds = FolhasRocole("val")
    probs, y, ids = prever(modelo, DataLoader(ds, batch_size=32), disp)
    pred_df = pd.DataFrame({"id": ids, "classe": [CLASSES_ROCOLE[k] for k in y],
                            "classe_predita": [CLASSES_ROCOLE[k] for k in probs.argmax(1)],
                            **{f"p_{c}": probs[:, k] for k, c in enumerate(CLASSES_ROCOLE)}})
    pred_df.to_csv(pasta / "predicoes_val.csv", index=False)
    grupos = grupos_por_id(ids, ds.meta)

    cv = deslocamentos_cv(probs, y, grupos)
    final = escolher_deslocamentos(probs, y)
    folhas = avaliar_folhas(pred_df)
    folhas.to_csv(pasta / "folhas_val.csv", index=False)
    saida = {"n_val": int(len(y)), "n_por_classe": {c: int((y == k).sum()) for k, c in enumerate(CLASSES_ROCOLE)},
             "cv_dentro_da_validacao": {k: v for k, v in cv.items() if k != "pred_fora_da_dobra"},
             "deslocamentos_finais": {c: float(v) for c, v in zip(CLASSES_ROCOLE, final)}}
    with open(pasta / "validacao_ajustes.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2)

    print(f"\n--- Deslocamentos por classe (validacao, {len(y)} fotos; ganho medido fora da dobra, 5 dobras por planta) ---")
    tab = pd.DataFrame({"argmax": {"F1 macro": cv["argmax"]["f1_macro"], "acuracia": cv["argmax"]["acuracia"],
                                   **{f"recall {NOMES_ROCOLE_PT[c]}": v for c, v in cv["argmax"]["recall"].items()}},
                        "ajustado": {"F1 macro": cv["ajustado"]["f1_macro"], "acuracia": cv["ajustado"]["acuracia"],
                                     **{f"recall {NOMES_ROCOLE_PT[c]}": v for c, v in cv["ajustado"]["recall"].items()}}})
    print(tab.round(4).to_string())
    print("deslocamentos (acaro, ferrugem) escolhidos em cada dobra:", cv["deslocamentos_por_dobra_acaro_ferrugem"])
    print("deslocamentos escolhidos na validacao inteira:", saida["deslocamentos_finais"])
    print(f"\n--- Folha inteira (validacao, {folhas['folhas'].iloc[0]} folhas com as duas fotos) ---")
    print(folhas.round(4).to_string(index=False))
    print(f"\nResultados salvos em: {pasta}")


if __name__ == "__main__":
    main()
