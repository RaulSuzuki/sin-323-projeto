"""Pontua o modelo de Hough do BRACOL no teste, UMA vez por modelo final.

Uso:
  python -P src/hough_avaliar.py --bracol   # passo 1 do aplicativo, saudavel ou doenca, no teste

No teste do BRACOL (201 folhas): o passo 1 do aplicativo. Doenca quando P(doenca) = 1 - P(saudavel) e a maior das
duas; "incerto" quando nenhuma passa de LIMITE_INCERTO (comum.py). Acerto, doencas achadas, saudaveis certas e
incertas, com IC 95% por bootstrap de grupos; e os mesmos numeros fora da dobra (cv_oof.csv), para comparar.
Configuracao e limiar vem so de treino + validacao. Para se o teste ja foi pontuado para aquele modelo (--refazer
passa por cima, de proposito).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import hough_treinar as ht
from comum import LIMITE_INCERTO, SEED
from hough_dados import adicionar_opcao_dataset, carregar_conjunto

CHAVES = ["sensibilidade", "especificidade", "acuracia_balanceada", "f1", "acuracia"]


def conferir_primeira_vez(arq, refazer=False):
    """O teste e pontuado uma vez por modelo final: para se o resultado ja existe."""
    if Path(arq).exists() and not refazer:
        raise SystemExit(f"{arq} ja existe: o teste ja foi pontuado para este modelo. Use --refazer so de proposito "
                         "(por exemplo, depois de treinar de novo).")


def metricas_binarias(y, pred):
    """Sensibilidade, especificidade, acuracia balanceada, F1 e precisao da classe positiva (doenca), e acuracia.
    Razoes sem denominador viram NaN (o bootstrap as ignora)."""
    y, p = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, tn = int((y & p).sum()), int((~y & ~p).sum())
    fp, fn = int((~y & p).sum()), int((y & ~p).sum())
    div = lambda a, b: a / b if b else float("nan")
    sens, espec = div(tp, tp + fn), div(tn, tn + fp)
    return {"sensibilidade": sens, "especificidade": espec, "acuracia_balanceada": (sens + espec) / 2,
            "f1": div(2 * tp, 2 * tp + fp + fn), "precisao": div(tp, tp + fp), "acuracia": div(tp + tn, len(y))}


def bootstrap_binario(y, pred, grupos, b=2000, seed=SEED):
    """IC 95% das metricas_binarias, reamostrando grupos (as quase-copias ficam juntas) com reposicao."""
    y, pred, grupos = np.asarray(y), np.asarray(pred), np.asarray(grupos)
    rng = np.random.default_rng(seed)
    unicos = np.unique(grupos)
    por_grupo = [np.flatnonzero(grupos == g) for g in unicos]
    amostras = []
    for _ in range(b):
        i = np.concatenate([por_grupo[k] for k in rng.integers(0, len(unicos), len(unicos))])
        m = metricas_binarias(y[i], pred[i])
        amostras.append([m[c] for c in CHAVES])
    boot = pd.DataFrame(amostras, columns=CHAVES)
    base = metricas_binarias(y, pred)
    return pd.DataFrame({"estimativa": pd.Series({c: base[c] for c in CHAVES}),
                         "IC95_inf": boot.quantile(0.025), "IC95_sup": boot.quantile(0.975)})


def passo_1(y_doenca, p_doenca, limite=LIMITE_INCERTO):
    """O passo 1 do aplicativo: doenca quando P(doenca) >= 0,5 (a maior das duas); incerto quando
    max(P(doenca), P(saudavel)) < limite. Devolve acerto, doencas achadas, saudaveis certas, fracao incerta e o
    acerto nas incertas e nas outras."""
    y, p = np.asarray(y_doenca, bool), np.asarray(p_doenca, float)
    pred = p >= 0.5
    incerto = np.maximum(p, 1 - p) < limite
    certo = pred == y
    taxa = lambda m: float(certo[m].mean()) if m.any() else None
    return {"n": int(len(y)), "acerto": float(certo.mean()), "doenca_achada": taxa(y), "saudavel_certa": taxa(~y),
            "incerto": float(incerto.mean()), "acerto_nas_incertas": taxa(incerto), "acerto_nas_outras": taxa(~incerto),
            "vp": int((y & pred).sum()), "fn": int((y & ~pred).sum()), "fp": int((~y & pred).sum()),
            "vn": int((~y & ~pred).sum())}


def _passo_1_fora_da_dobra(conj, modelo):
    """passo_1 nas predicoes fora da dobra do modelo (cv_oof.csv, de hough_treinar.py cv), se o arquivo existe."""
    arq = conj.pasta / "cv_oof.csv"
    if not arq.exists():
        return None
    oof = pd.read_csv(arq)
    saudavel = list(modelo["classes"])[0]
    return passo_1(oof["classe"] != saudavel, 1 - oof[f"p_{saudavel}"])


def pontuar_em(conj, refazer=False):
    """Pontua, no TESTE do conjunto `conj` (o BRACOL), o modelo de Hough treinado nele (conj.artefato): o passo 1 do
    aplicativo, saudavel ou doenca. Grava resultados_teste.json, predicoes_teste.csv e teste_intervalos_confianca.csv
    na pasta de `conj`, uma vez."""
    arq = conj.pasta / "resultados_teste.json"
    conferir_primeira_vez(arq, refazer)
    if not conj.artefato.exists():
        raise SystemExit(f"{conj.artefato} nao existe: rode hough_treinar.py final --dataset {conj.nome}")
    modelo = joblib.load(conj.artefato)
    if modelo.get("metadados_sha256") != conj.sha256():
        raise SystemExit(f"o modelo de '{conj.nome}' foi treinado com outra divisao: rode hough_treinar.py final "
                         f"--dataset {conj.nome} de novo")
    te = conj.meta[conj.meta["split"] == "test"].reset_index(drop=True)
    dados = ht.atributos_do_conjunto(conj)
    p = ht.prever(modelo, te["id"], dados)
    classes = list(modelo["classes"])
    saudavel = classes[0]
    p_doenca = 1 - p[:, 0]
    y, pred, grupos = (te["classe"] != saudavel).to_numpy(), p_doenca >= 0.5, te["grupo"].to_numpy()
    teste = passo_1(y, p_doenca)
    ic = bootstrap_binario(y, pred, grupos)  # sensibilidade = doencas achadas; especificidade = saudaveis certas
    nomes_ic = {"acuracia": "acerto", "sensibilidade": "doenca_achada", "especificidade": "saudavel_certa"}
    resultados = {"conjunto": conj.nome, "n_teste": int(len(te)), "n_doenca": int(y.sum()), "classes": classes,
                  "limite_incerto": LIMITE_INCERTO,
                  "regra": "doenca quando P(doenca) = 1 - P(saudavel) e a maior das duas; incerto quando nenhuma passa "
                           f"de {LIMITE_INCERTO:.0%}",
                  "teste": teste,
                  "ic95": {nomes_ic[m]: [float(ic.loc[m, "IC95_inf"]), float(ic.loc[m, "IC95_sup"])] for m in nomes_ic},
                  "fora_da_dobra": _passo_1_fora_da_dobra(conj, modelo)}
    conj.pasta.mkdir(parents=True, exist_ok=True)
    te.assign(**{f"p_{c}": p[:, j] for j, c in enumerate(classes)}, doenca=pred,
              incerto=np.maximum(p_doenca, 1 - p_doenca) < LIMITE_INCERTO)[
        ["id", "classe", "grupo"] + [f"p_{c}" for c in classes] + ["doenca", "incerto"]].to_csv(
        conj.pasta / "predicoes_teste.csv", index=False)
    ic.to_csv(conj.pasta / "teste_intervalos_confianca.csv")
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)
    print(f"--- Teste de '{conj.nome}' ({len(te)} folhas, {int(y.sum())} com doenca): passo 1, saudavel ou doenca ---")
    print(f"acerto {teste['acerto']:.3f} (IC 95% {resultados['ic95']['acerto'][0]:.3f} a "
          f"{resultados['ic95']['acerto'][1]:.3f}) | doencas achadas {teste['vp']} de {teste['vp'] + teste['fn']} | "
          f"saudaveis certas {teste['vn']} de {teste['vn'] + teste['fp']} | incertas {teste['incerto']:.1%}")
    if resultados["fora_da_dobra"]:
        f = resultados["fora_da_dobra"]
        print(f"fora da dobra (treino + validacao): acerto {f['acerto']:.3f} | incertas {f['incerto']:.1%} (acerto "
              f"{f['acerto_nas_incertas'] or 0:.0%} nelas)")
    return resultados


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    adicionar_opcao_dataset(ap)
    ap.add_argument("--refazer", action="store_true", help="pontua o teste de novo, de proposito")
    args = ap.parse_args(argv)
    pontuar_em(carregar_conjunto(args.dataset), refazer=args.refazer)


if __name__ == "__main__":
    main()
