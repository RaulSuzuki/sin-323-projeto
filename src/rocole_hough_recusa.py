"""Opcao de recusa do detector de ferrugem (fusao Hough + ResNet-50): "nao sei, mostre a um agronomo".

Uso (na raiz do repositorio, com PYTHONPATH=src):
  python -P src/rocole_hough_recusa.py cv      # escolhe a taxa de recusa em treino + validacao (aninhado)
  python -P src/rocole_hough_recusa.py teste   # pontua o ponto escolhido no teste, UMA vez (predicoes ja gravadas)

A foto e recusada quando P(ferrugem) cai numa faixa em volta do limiar: |P - limiar| < margem. A margem e a que
recusa uma dada taxa das fotos. Tudo e escolhido fora da dobra: para cada dobra, o limiar e a margem vem das OUTRAS
duas (rocole_hough_treinar.escolher_limiar e margem_para_taxa).

Regra fixada antes de rodar: fica a MENOR taxa de recusa (de TAXAS) cujo F1 de ferrugem nas fotos aceitas, media
das 3 dobras, chega a META_F1. Se nenhuma chega, a recusa nao e adotada e nada e pontuado no teste.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import rocole_hough_treinar as ht
from rocole_comum import SEED

TAXAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
META_F1 = 0.90
PASTA = ht.PASTA
RESULTADOS = PASTA / "recusa_teste.json"


def margem_para_taxa(p, limiar, taxa):
    """Meia largura da faixa em volta do limiar que recusa a fracao `taxa` das fotos: quantil de |p - limiar|."""
    if taxa <= 0:
        return 0.0
    return float(np.quantile(np.abs(np.asarray(p, float) - limiar), taxa))


def decidir(p, limiar, margem):
    """1 = ferrugem, 0 = sem ferrugem, -1 = recusada (dentro da faixa)."""
    p = np.asarray(p, float)
    d = (p >= limiar).astype(int)
    d[np.abs(p - limiar) < margem] = -1
    return d


def metricas_com_recusa(y, p, limiar, margem):
    """Fracao aceita e, SO nas fotos aceitas, acertos/erros, F1, sensibilidade e especificidade de ferrugem;
    'recusadas_com_ferrugem' = fracao das recusadas que tinham ferrugem."""
    y, d = np.asarray(y, bool), decidir(p, limiar, margem)
    ac = d >= 0
    ya, pa = y[ac], d[ac] == 1
    vp, fn = int((ya & pa).sum()), int((ya & ~pa).sum())
    fp, vn = int((~ya & pa).sum()), int((~ya & ~pa).sum())
    div = lambda a, b: a / b if b else float("nan")
    return {"aceitas": float(ac.mean()), "n": int(len(y)), "n_recusadas": int((~ac).sum()), "vp": vp, "fn": fn,
            "fp": fp, "vn": vn, "f1": 2 * vp / (2 * vp + fp + fn) if vp else 0.0, "sensibilidade": div(vp, vp + fn),
            "especificidade": div(vn, vn + fp), "recusadas_com_ferrugem": div(int(y[~ac].sum()), int((~ac).sum()))}


def curva_aninhada(oof, taxas=TAXAS):
    """Para cada taxa de recusa e cada dobra: limiar e margem escolhidos nas outras dobras, metricas na dobra.
    oof: dobra, classe, p_rust (predicoes fora da dobra). Linhas: taxa, dobra, limiar, margem + metricas_com_recusa."""
    y, p, d = (oof["classe"] == "rust").to_numpy(), oof["p_rust"].to_numpy(float), oof["dobra"].to_numpy()
    linhas = []
    for k in sorted(np.unique(d)):
        fora = d != k
        limiar = ht.escolher_limiar(y[fora], p[fora])
        for taxa in taxas:
            margem = margem_para_taxa(p[fora], limiar, taxa)
            linhas.append({"taxa": taxa, "dobra": int(k), "limiar": limiar, "margem": margem,
                           **metricas_com_recusa(y[~fora], p[~fora], limiar, margem)})
    return pd.DataFrame(linhas).sort_values(["taxa", "dobra"]).reset_index(drop=True)


def resumo_curva(curva):
    """Por taxa: fracao aceita (media) e F1 nas aceitas (media, menor e maior dobra)."""
    g = curva.groupby("taxa")
    return pd.DataFrame({"aceitas_media": g["aceitas"].mean(), "f1_media": g["f1"].mean(), "f1_min": g["f1"].min(),
                         "f1_max": g["f1"].max()}).reset_index()


def escolher_taxa(resumo, meta=META_F1):
    """A menor taxa de recusa cujo F1 medio nas aceitas chega a meta; None se nenhuma chega."""
    ok = resumo[resumo["f1_media"] >= meta - 1e-9]
    return float(ok["taxa"].min()) if len(ok) else None


def bootstrap_recusa(y, p, grupos, limiar, margem, b=2000, seed=SEED):
    """IC 95% da fracao aceita e do F1 nas aceitas, reamostrando grupos (plantas) com reposicao."""
    y, p, grupos = np.asarray(y, bool), np.asarray(p, float), np.asarray(grupos)
    rng = np.random.default_rng(seed)
    unicos = np.unique(grupos)
    por_grupo = [np.flatnonzero(grupos == g) for g in unicos]
    amostras = []
    for _ in range(b):
        i = np.concatenate([por_grupo[k] for k in rng.integers(0, len(unicos), len(unicos))])
        m = metricas_com_recusa(y[i], p[i], limiar, margem)
        amostras.append([m["aceitas"], m["f1"]])
    boot = pd.DataFrame(amostras, columns=["aceitas", "f1"])
    base = metricas_com_recusa(y, p, limiar, margem)
    return pd.DataFrame({"estimativa": pd.Series({c: base[c] for c in boot.columns}),
                         "IC95_inf": boot.quantile(0.025), "IC95_sup": boot.quantile(0.975)})


def conferir_primeira_vez(arq=RESULTADOS, refazer=False):
    if Path(arq).exists() and not refazer:
        raise SystemExit(f"{arq} ja existe: a recusa ja foi pontuada no teste. Use --refazer so de proposito.")


def _decisao():
    arq = PASTA / "decisao.json"
    if not arq.exists():
        raise SystemExit(f"{arq} nao existe: rode rocole_hough_treinar.py cv, final e fusao")
    with open(arq, encoding="utf-8") as f:
        return arq, json.load(f)


def rodar_cv():
    """Escolhe a taxa de recusa fora da dobra; grava recusa_cv.csv e recusa_resumo.csv e registra a decisao em
    decisao.json e no modelo gravado (sem ajusta-lo de novo)."""
    arq, dec = _decisao()
    if not dec.get("fusao", {}).get("adotada") or not (PASTA / "fusao_oof.csv").exists():
        raise SystemExit("a recusa e definida sobre a fusao: rode rocole_hough_treinar.py fusao (e ela precisa ser adotada)")
    oof = pd.read_csv(PASTA / "fusao_oof.csv")
    curva = curva_aninhada(oof)
    resumo = resumo_curva(curva)
    curva.to_csv(PASTA / "recusa_cv.csv", index=False)
    resumo.to_csv(PASTA / "recusa_resumo.csv", index=False)
    taxa = escolher_taxa(resumo)
    limiar = dec["fusao"]["limiar"]
    recusa = {"adotada": taxa is not None and taxa > 0, "meta_f1": META_F1, "taxas_testadas": list(TAXAS),
              "regra": "a menor taxa de recusa cujo F1 de ferrugem nas fotos aceitas (fora da dobra, media das 3 dobras, "
                       "limiar e margem das outras dobras) chega a meta"}
    if taxa is not None:
        linha = resumo.set_index("taxa").loc[taxa]
        # margem do modelo final: a que recusa essa taxa de TODAS as predicoes fora da dobra, com o limiar da fusao
        recusa.update({"taxa": taxa, "limiar": limiar, "margem": margem_para_taxa(oof["p_rust"], limiar, taxa),
                       "f1_media_aninhada": float(linha["f1_media"]), "aceitas_media": float(linha["aceitas_media"])})
    dec["recusa"] = recusa
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(dec, f, indent=2)
    if ht.ARTEFATO.exists():
        modelo = joblib.load(ht.ARTEFATO)
        modelo["recusa"] = ht._recusa_do_artefato(recusa)
        joblib.dump(modelo, ht.ARTEFATO, compress=3)
    print("--- Recusa, fora da dobra (fusao; limiar e margem de cada dobra escolhidos nas outras) ---")
    print(resumo.round(4).to_string(index=False))
    if recusa["adotada"]:
        print(f"\nescolhida: recusar {taxa:.0%} | margem {recusa['margem']:.3f} em volta do limiar {limiar:.2f} "
              f"(recusa se {limiar - recusa['margem']:.3f} < P < {limiar + recusa['margem']:.3f})")
    else:
        print(f"\nnenhuma taxa chega a F1 {META_F1:.2f} nas aceitas (ou a taxa 0 ja chega): recusa nao adotada")
    return recusa


def pontuar_teste(refazer=False):
    """O ponto escolhido, no teste, uma vez: le as probabilidades da fusao ja gravadas (nao roda modelo nenhum)."""
    conferir_primeira_vez(refazer=refazer)
    _, dec = _decisao()
    recusa = dec.get("recusa") or {}
    if not recusa.get("adotada"):
        raise SystemExit("a recusa nao foi adotada (rode rocole_hough_recusa.py cv): nada a pontuar")
    arq_pred = PASTA / "predicoes_teste_fusao.csv"
    if not arq_pred.exists():
        raise SystemExit(f"{arq_pred} nao existe: rode rocole_hough_avaliar.py --fusao")
    te = pd.read_csv(arq_pred)
    y, p, limiar, margem = (te["classe"] == "rust").to_numpy(), te["p_rust"].to_numpy(), recusa["limiar"], recusa["margem"]
    m, sem = metricas_com_recusa(y, p, limiar, margem), metricas_com_recusa(y, p, limiar, 0.0)
    ic = bootstrap_recusa(y, p, te["grupo"].to_numpy(), limiar, margem)
    d = decidir(p, limiar, margem)
    por_classe = te.assign(recusada=d < 0).groupby("classe")["recusada"].agg(["sum", "count"])
    resultados = {"taxa_escolhida": recusa["taxa"], "limiar": limiar, "margem": margem, "com_recusa": m, "sem_recusa": sem,
                  "ic95": {c: {"inf": float(ic.loc[c, "IC95_inf"]), "sup": float(ic.loc[c, "IC95_sup"])} for c in ic.index},
                  "recusadas_por_classe": {c: [int(r["sum"]), int(r["count"])] for c, r in por_classe.iterrows()}}
    te.assign(decisao=np.where(d < 0, "recusada", np.where(d == 1, "ferrugem", "sem ferrugem")))[
        ["id", "classe", "lado", "grupo", "p_rust", "decisao"]].to_csv(PASTA / "recusa_teste_predicoes.csv", index=False)
    ic.to_csv(PASTA / "recusa_teste_ic.csv")
    with open(RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)
    print(f"--- Recusa no teste ({len(te)} fotos; recusa se |P - {limiar:.2f}| < {margem:.3f}) ---")
    print(f"aceitas: {m['aceitas']:.1%} ({m['n'] - m['n_recusadas']} de {m['n']}) | nas aceitas: F1 {m['f1']:.4f}, "
          f"sensibilidade {m['sensibilidade']:.4f}, especificidade {m['especificidade']:.4f} "
          f"(vp {m['vp']}, fn {m['fn']}, fp {m['fp']}, vn {m['vn']})")
    print(f"sem recusa (todas as fotos): F1 {sem['f1']:.4f}")
    print("IC 95% por bootstrap de plantas:")
    print(ic.round(4).to_string())
    print("recusadas por classe (recusadas, total):")
    print(por_classe.to_string())
    return resultados


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("etapa", choices=["cv", "teste"])
    ap.add_argument("--refazer", action="store_true", help="pontua o teste de novo, de proposito")
    args = ap.parse_args(argv)
    if args.etapa == "cv":
        rodar_cv()
    else:
        pontuar_teste(args.refazer)


if __name__ == "__main__":
    main()
