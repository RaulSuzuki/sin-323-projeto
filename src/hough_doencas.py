"""Modelo das doencas do BRACOL: uma probabilidade por doenca, para a etapa 2 do aplicativo (folha com doenca).

O modelo de Hough do BRACOL (hough_treinar.py) responde saudavel, outra doenca ou ferrugem: bicho-mineiro, mancha de
phoma e cercosporiose ficam juntas em "outra doenca". Este modelo separa as quatro doencas. Uma folha pode ter mais de
uma (os rotulos do BRACOL sao 0/1 por doenca), entao e um classificador por doenca (multirrotulo), cada um com o seu
limiar.

Os atributos sao os do modelo de Hough (hough_treinar.montar_x, na variante do modelo gravado), com as pontuacoes dos
circulos pelo nivel 1. No treino, essas pontuacoes sao fora da dobra, por grupo, como no nivel 2 do modelo de Hough.
Treino e avaliacao usam so as folhas com pelo menos uma doenca: a etapa 1 do aplicativo (saudavel ou doenca) continua
com o modelo de Hough.

Uso (na raiz do repositorio, com PYTHONPATH=src). Precisa do modelo de Hough do BRACOL (models/bracol_hough.joblib) e
do cache de atributos do BRACOL (hough_treinar.py extrair --bracol):
  python -P src/hough_doencas.py cv        # validacao cruzada no treino + validacao (as 3 dobras do modelo de Hough):
                                            # escolhe o classificador e o limiar de cada doenca (a base)
  python -P src/hough_doencas.py comparar  # so na validacao cruzada: + nivel 1 por doenca (5 classes), contra a base
  python -P src/hough_doencas.py adotar    # se o nivel 1 por doenca passou a regra: nova decisao
  python -P src/hough_doencas.py final     # ajusta em treino + validacao e grava models/bracol_doencas.joblib
  python -P src/hough_doencas.py teste     # pontua o teste UMA vez (para se doencas_teste.json ja existe)

Tudo e escolhido no treino + validacao; o teste so e lido pelo comando teste. Resultados em outputs/hough_bracol/
(doencas_*). Sem models/bracol_doencas.joblib, o aplicativo mostra as classes do modelo de Hough na etapa 2.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

import hough as rh
import hough_treinar as ht
from comum import MODELOS_ROCOLE, SEED
from hough_dados import DOENCAS_BRACOL, carregar_conjunto

DOENCAS = list(DOENCAS_BRACOL)  # miner, rust, phoma, cercospora
TIPOS = ("floresta", "hgb", "logistica")  # classificadores comparados na validacao cruzada; o primeiro e a base
ARTEFATO = MODELOS_ROCOLE / "bracol_doencas.joblib"
N_BOOTSTRAP = 1000


# ---------------------------------------------------------------- modelo
def com_doenca(meta):
    """As folhas de meta (colunas DOENCAS) com pelo menos uma doenca."""
    return meta[meta[DOENCAS].sum(axis=1) > 0]


def ajustar_doencas(x, y, tipo="floresta"):
    """Um classificador binario por doenca (hough_treinar._nivel2(tipo)). x: (n, k); y: (n, len(DOENCAS)) 0/1.
    Doenca sem exemplos positivos e negativos no treino: um numero (a fracao de positivos), usado como probabilidade."""
    y = np.asarray(y, int)
    return [float(y[:, j].mean()) if len(np.unique(y[:, j])) < 2 else ht._nivel2(tipo).fit(x, y[:, j])
            for j in range(y.shape[1])]


def prever_doencas(modelos, x):
    """(n, len(modelos)): probabilidade de cada doenca."""
    p = np.zeros((len(x), len(modelos)))
    for j, m in enumerate(modelos):
        p[:, j] = m if isinstance(m, float) else m.predict_proba(x)[:, list(m.classes_).index(1)]
    return p


def prever_foto(modelo, dados_foto, p_circ):
    """Uma foto ({"circulos", "at_circulos", "at_folha"}) e as pontuacoes (N, 3) dos seus circulos pelo nivel 1 do
    modelo de Hough -> {doenca: probabilidade}. Usado pelo aplicativo (hough_inferir.py)."""
    x = ht.montar_x(["foto"], {"foto": dados_foto}, {"foto": np.asarray(p_circ)}, modelo["variante"])
    if modelo.get("nivel1_doencas") is not None:  # o resumo do nivel 1 por doenca, como no treino
        dados = {"foto": dados_foto}
        x = np.hstack([x, x_por_doenca(["foto"], dados, pontuar5(modelo["nivel1_doencas"], ["foto"], dados))])
    return {d: float(v) for d, v in zip(modelo["doencas"], prever_doencas(modelo["modelos"], x)[0])}


# ---------------------------------------------------------------- nivel 1 por doenca (experimento: hough_doencas.py comparar)
# O nivel 1 do modelo de Hough so conhece saudavel / outra doenca / ferrugem, e os atributos "ricos" resumem os 5
# circulos mais parecidos com ferrugem: numa folha com ferrugem e outra doenca, a outra some na media. Aqui, um nivel 1
# com as 5 classes (saudavel e as 4 doencas), treinado so com circulos de folhas de rotulo unico (onde o rotulo fraco
# "o circulo tem a doenca da folha" vale), e um resumo por classe: pontuacoes e como sao os 5 circulos mais parecidos
# com cada uma.
CLASSES5 = ["healthy"] + DOENCAS
NOMES_POR_DOENCA = ["n_circulos"] + [f"{c}_{e}" for c in CLASSES5 for e in
                                     ["max", "top3", "top5", "media", "soma", "n_05", "frac_maior", "area"]
                                     + [f"top5_{a}" for a in ht.ATRIBUTOS_RICOS]]


def rotulo_unico(meta):
    """Classe de cada folha para o nivel 1 por doenca (indice = o de meta): 'healthy' sem doenca, a doenca quando e uma
    so, None quando a folha tem mais de uma (seus circulos nao entram no treino do nivel 1)."""
    n = meta[DOENCAS].sum(axis=1)
    rot = pd.Series([None] * len(meta), index=meta.index, dtype=object)
    rot[n == 0] = "healthy"
    for d in DOENCAS:
        rot[(n == 1) & (meta[d] == 1)] = d
    return rot


def _circulos_rotulados(ids, rot, dados):
    """Atributos e rotulo fraco (indice em CLASSES5) dos circulos das folhas de ids com rotulo unico."""
    ids = [i for i in ids if rot[i] is not None and len(dados[i]["at_circulos"])]
    x = np.vstack([dados[i]["at_circulos"] for i in ids])
    y = np.concatenate([np.full(len(dados[i]["at_circulos"]), CLASSES5.index(rot[i])) for i in ids])
    return x, y


def pontuar5(modelo, ids, dados):
    """id -> (N, 5): probabilidade de cada circulo ser de cada classe de CLASSES5."""
    saida = {}
    for i in ids:
        at = dados[i]["at_circulos"]
        p = np.zeros((len(at), len(CLASSES5)))
        if len(at):
            p[:, modelo.classes_] = modelo.predict_proba(at)
        saida[i] = p
    return saida


def nivel1_por_doenca(base, dados, n_dobras=ht.N_INTERNAS):
    """base: folhas de treino (indice = id; colunas DOENCAS e grupo). Devolve o nivel 1 por doenca ajustado em todas as
    folhas de rotulo unico de base e as pontuacoes FORA DA DOBRA (por grupo) dos circulos de todas as folhas de base,
    que e o que o classificador das doencas pode ver no treino."""
    from sklearn.model_selection import StratifiedGroupKFold
    rot = rotulo_unico(base)
    ids = np.asarray(base.index)
    sgkf = StratifiedGroupKFold(n_splits=n_dobras, shuffle=True, random_state=SEED)
    p = {}
    for tr, va in sgkf.split(ids, rot.fillna("mista").to_numpy(), base["grupo"].to_numpy()):
        p.update(pontuar5(ht._nivel1().fit(*_circulos_rotulados(ids[tr], rot, dados)), ids[va], dados))
    return ht._nivel1().fit(*_circulos_rotulados(ids, rot, dados)), p


def estatisticas_por_doenca(p, at_circulos, raios, area_folha=rh.LADO_FOLHA ** 2):
    """Resumo dos circulos de uma foto, classe a classe (ordem de NOMES_POR_DOENCA): quantos circulos; para cada classe
    de CLASSES5, maximo, media dos 3 e dos 5 maiores, media e soma de P(classe), quantos passam de 0,5, fracao dos
    circulos em que ela e a mais provavel, area pesada por P(classe) e a media de ATRIBUTOS_RICOS nos 5 circulos mais
    parecidos com ela. Sem circulos: zeros."""
    p = np.asarray(p, np.float64).reshape(-1, len(CLASSES5))
    if not len(p):
        return np.zeros(len(NOMES_POR_DOENCA), np.float32)
    at = np.asarray(at_circulos, np.float64)[:, ht._IDX_RICOS]
    r = np.asarray(raios, np.float64)
    maior = p.argmax(1)
    saida = [len(p)]
    for c in range(len(CLASSES5)):
        o = np.sort(p[:, c])[::-1]
        top = np.argsort(-p[:, c], kind="stable")[:5]
        saida += [o[0], o[:3].mean(), o[:5].mean(), o.mean(), o.sum(), (p[:, c] > 0.5).sum(), (maior == c).mean(),
                  (p[:, c] * np.pi * r ** 2).sum() / area_folha]
        saida += list(at[top].mean(0))
    return np.asarray(saida, np.float32)


def x_por_doenca(ids, dados, p5):
    return np.stack([estatisticas_por_doenca(p5[i], dados[i]["at_circulos"], dados[i]["circulos"][:, 2]) for i in ids])


# ---------------------------------------------------------------- validacao cruzada
def oof_doencas(meta, dados, variante, nomes, tipos=TIPOS, por_doenca=False):
    """Probabilidades fora da dobra das folhas com doenca do treino + validacao, para cada tipo de classificador:
    DataFrame id, dobra, grupo, y_<doenca>, p_<tipo>_<doenca>. As dobras sao as do modelo de Hough
    (hough_treinar.dobras). O nivel 1 de cada dobra e ajustado nas folhas de treino da dobra, saudaveis tambem, como
    no modelo de Hough; as folhas de treino sao pontuadas fora da dobra, e as de fora, pelo nivel 1 da dobra.
    por_doenca: junta aos atributos o resumo do nivel 1 por doenca (nivel1_por_doenca), com o mesmo cuidado."""
    d = ht.dobras(meta)
    base = meta[meta["split"].isin(["train", "val"])].set_index("id").loc[d.index]
    partes = []
    for k in range(ht.N_DOBRAS):
        tr, fora = base[(d != k).to_numpy()], base[(d == k).to_numpy()]
        n1, p_tr = ht.ajustar_nivel1(tr.index, tr["classe"], tr["grupo"], dados, nomes)
        tr_d, fora_d = com_doenca(tr), com_doenca(fora)
        x_tr = ht.montar_x(list(tr_d.index), dados, p_tr, variante)
        x_fora = ht.montar_x(list(fora_d.index), dados, ht._pontuar(n1, list(fora_d.index), dados), variante)
        if por_doenca:
            n1_5, p5_tr = nivel1_por_doenca(tr, dados)
            x_tr = np.hstack([x_tr, x_por_doenca(list(tr_d.index), dados, p5_tr)])
            x_fora = np.hstack([x_fora, x_por_doenca(list(fora_d.index), dados, pontuar5(n1_5, list(fora_d.index), dados))])
        linha = {"id": fora_d.index.to_numpy(), "dobra": k, "grupo": fora_d["grupo"].to_numpy(),
                 **{f"y_{dz}": fora_d[dz].to_numpy(int) for dz in DOENCAS}}
        for tipo in tipos:
            p = prever_doencas(ajustar_doencas(x_tr, tr_d[DOENCAS].to_numpy(int), tipo), x_fora)
            linha.update({f"p_{tipo}_{dz}": p[:, j] for j, dz in enumerate(DOENCAS)})
        partes.append(pd.DataFrame(linha))
    return pd.concat(partes, ignore_index=True)


def escolher_limiares(y, p):
    """Limiar de cada doenca com o maior F1 (hough_treinar.escolher_limiar). y, p: (n, len(DOENCAS))."""
    return np.array([ht.escolher_limiar(y[:, j], p[:, j]) for j in range(y.shape[1])])


def metricas(y, p, limiares):
    """y, p: (n, len(DOENCAS)). Tabela por doenca (positivos, precisao, recall, F1, AP) e o resumo: F1 macro, acerto
    exato do conjunto de doencas da folha e acerto da doenca mais provavel (a de maior p esta entre as reais)."""
    from sklearn.metrics import average_precision_score
    y, p = np.asarray(y, bool), np.asarray(p, float)
    pred = p >= np.asarray(limiares)
    linhas = []
    for j, dz in enumerate(DOENCAS):
        yj, pj = y[:, j], pred[:, j]
        tp = int((yj & pj).sum())
        linhas.append({"doenca": dz, "positivos": int(yj.sum()), "precisao": tp / max(int(pj.sum()), 1),
                       "recall": tp / max(int(yj.sum()), 1), "f1": ht._f1(yj, pj),
                       "ap": float(average_precision_score(yj, p[:, j])) if yj.any() and (~yj).any() else float("nan")})
    tab = pd.DataFrame(linhas)
    top = y[np.arange(len(y)), p.argmax(1)] if len(y) else np.zeros(0, bool)
    return tab, {"n": int(len(y)), "f1_macro": float(tab["f1"].mean()), "acerto_exato": float((pred == y).all(1).mean()),
                 "acerto_mais_provavel": float(top.mean()) if len(top) else float("nan")}


def _yp(oof, tipo):
    return oof[[f"y_{dz}" for dz in DOENCAS]].to_numpy(int), oof[[f"p_{tipo}_{dz}" for dz in DOENCAS]].to_numpy()


def f1_macro_por_dobra(oof, tipo):
    """F1 macro de cada dobra, com os limiares escolhidos nas OUTRAS dobras (o numero honesto)."""
    y, p = _yp(oof, tipo)
    d = oof["dobra"].to_numpy()
    return pd.Series({int(k): metricas(y[d == k], p[d == k], escolher_limiares(y[d != k], p[d != k]))[1]["f1_macro"]
                      for k in sorted(np.unique(d))}, name=tipo)


def escolher_tipo(oof, tipos=TIPOS):
    """Regra fixada antes de rodar (a mesma do modelo de Hough, hough_treinar.decidir): um classificador substitui a
    base (tipos[0]) so se a media do F1 macro for maior E se vencer em pelo menos 2 das 3 dobras; entre os que passam,
    o de maior media. Devolve (tipo escolhido, tabela dobras x tipos)."""
    tab = pd.concat([f1_macro_por_dobra(oof, t) for t in tipos], axis=1)
    passam = [t for t in tipos[1:] if ht.decidir(tab, tipos[0], t)]
    return (max(passam, key=lambda t: tab[t].mean()) if passam else tipos[0]), tab


def pred_honesta(oof, tipo):
    """(y, predicao) 0/1 fora da dobra, com os limiares de cada dobra escolhidos nas OUTRAS dobras."""
    y, p = _yp(oof, tipo)
    d = oof["dobra"].to_numpy()
    pred = np.zeros(y.shape, bool)
    for k in np.unique(d):
        pred[d == k] = p[d == k] >= escolher_limiares(y[d != k], p[d != k])
    return y.astype(bool), pred


def tabela_por_doenca(oof, tipo):
    """Por doenca, com pred_honesta: F1, recall nas folhas so com ela e nas folhas com mais de uma doenca, e alarmes."""
    y, pred = pred_honesta(oof, tipo)
    mista = y.sum(1) > 1

    def recall(m):
        return float((pred_j & m).sum() / max(int(m.sum()), 1))

    linhas = []
    for j, dz in enumerate(DOENCAS):
        yj, pred_j = y[:, j], pred[:, j]
        linhas.append({"doenca": dz, "f1": ht._f1(yj, pred_j), "recall_so_ela": recall(yj & ~mista),
                       "recall_com_outra": recall(yj & mista), "alarmes": int((~yj & pred_j).sum())})
    return pd.DataFrame(linhas)


def comparar(conj, hough=None, dados=None, tipos=TIPOS):
    """Experimento, so na validacao cruzada (o teste nao e lido): os atributos atuais + o resumo do nivel 1 por doenca,
    contra o modelo atual (doencas_cv_oof.csv, as mesmas dobras). Mesma regra de sempre: a alternativa substitui o atual
    so com media do F1 macro maior E vitoria em pelo menos 2 das 3 dobras. Grava doencas_cv_por_doenca_oof.csv,
    doencas_cv_por_doenca_resumo.csv e doencas_cv_por_doenca.json; nao mexe no modelo nem em doencas_decisao.json."""
    arq_oof, arq_dec = conj.pasta / "doencas_cv_oof.csv", conj.pasta / "doencas_decisao.json"
    if not arq_oof.exists() or not arq_dec.exists():
        raise SystemExit(f"{arq_oof} ou {arq_dec} nao existe: rode hough_doencas.py cv")
    atual = pd.read_csv(arq_oof, dtype={"id": str})
    with open(arq_dec, encoding="utf-8") as f:
        tipo_atual = json.load(f)["tipo"]
    if hough is None:
        hough, dados = _contexto(conj)
    nova = oof_doencas(conj.meta, dados, hough["variante"], conj.classes, tipos, por_doenca=True)
    if sorted(nova["id"].astype(str)) != sorted(atual["id"]):
        raise SystemExit("as folhas da validacao cruzada nao batem com as de doencas_cv_oof.csv")
    base = f"atual_{tipo_atual}"
    tab = pd.concat([f1_macro_por_dobra(atual, tipo_atual).rename(base)]
                    + [f1_macro_por_dobra(nova, t).rename(f"por_doenca_{t}") for t in tipos], axis=1)
    passam = [c for c in tab.columns[1:] if ht.decidir(tab, base, c)]
    escolhida = max(passam, key=lambda c: tab[c].mean()) if passam else base
    melhor = max(tab.columns[1:], key=lambda c: tab[c].mean())
    t_atual = tabela_por_doenca(atual, tipo_atual).assign(modelo=base)
    t_nova = tabela_por_doenca(nova, melhor.removeprefix("por_doenca_")).assign(modelo=melhor)
    conj.pasta.mkdir(parents=True, exist_ok=True)
    nova.to_csv(conj.pasta / "doencas_cv_por_doenca_oof.csv", index=False)
    tab.rename_axis("dobra").to_csv(conj.pasta / "doencas_cv_por_doenca_resumo.csv")
    res = {"f1_macro_por_dobra": {c: tab[c].round(4).tolist() for c in tab}, "media": tab.mean().round(4).to_dict(),
           "escolhida": escolhida, "substitui_o_atual": escolhida != base,
           "por_doenca": pd.concat([t_atual, t_nova]).round(4).to_dict("records"),
           "regra": "substitui o atual so com media maior E vitoria em >= 2 das 3 dobras; o teste nao foi lido"}
    with open(conj.pasta / "doencas_cv_por_doenca.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(tab.round(4).to_string())
    print(f"\nmedias: {res['media']}\nescolhida pela regra: {escolhida} (substitui o atual: {res['substitui_o_atual']})")
    print(pd.concat([t_atual, t_nova]).round(3).to_string(index=False))
    return res


# ---------------------------------------------------------------- comandos
def _contexto(conj):
    """Modelo de Hough gravado (variante, parametros do Hough) e os atributos do cache do conjunto."""
    if not conj.artefato.exists():
        raise SystemExit(f"{conj.artefato} nao existe: rode hough_treinar.py cv e final --dataset {conj.nome}")
    hough = joblib.load(conj.artefato)
    return hough, ht.atributos_do_conjunto(conj, hough["param1"], hough["param2"])


def rodar_cv(conj, hough=None, dados=None, tipos=TIPOS):
    """Validacao cruzada e escolha do classificador e dos limiares. Grava doencas_cv_oof.csv, doencas_cv_resumo.csv e
    doencas_decisao.json em conj.pasta."""
    if hough is None:
        hough, dados = _contexto(conj)
    oof = oof_doencas(conj.meta, dados, hough["variante"], conj.classes, tipos)
    tipo, tab = escolher_tipo(oof, tipos)
    y, p = _yp(oof, tipo)
    limiares = escolher_limiares(y, p)
    por_doenca, resumo = metricas(y, p, limiares)
    conj.pasta.mkdir(parents=True, exist_ok=True)
    oof.to_csv(conj.pasta / "doencas_cv_oof.csv", index=False)
    tab.rename_axis("dobra").to_csv(conj.pasta / "doencas_cv_resumo.csv")
    dec = {"tipo": tipo, "limiares": dict(zip(DOENCAS, map(float, limiares))), "variante": hough["variante"],
           "f1_macro_por_dobra": {t: tab[t].round(4).tolist() for t in tab}, "fora_da_dobra": resumo,
           "por_doenca": por_doenca.round(4).to_dict("records"),
           "regra": "o classificador substitui a base so com media maior E vitoria em >= 2 das 3 dobras"}
    with open(conj.pasta / "doencas_decisao.json", "w", encoding="utf-8") as f:
        json.dump(dec, f, indent=2, ensure_ascii=False)
    print(tab.round(4).to_string())
    print(f"\nescolhido: {tipo} | limiares {dec['limiares']}")
    print(por_doenca.round(3).to_string(index=False))
    print(f"fora da dobra ({resumo['n']} folhas com doenca): F1 macro {resumo['f1_macro']:.3f}, acerto exato "
          f"{resumo['acerto_exato']:.3f}, doenca mais provavel certa {resumo['acerto_mais_provavel']:.3f}")
    return dec


def adotar(conj):
    """Adota o resultado de comparar, se a alternativa passou a regra: escolhe o limiar de cada doenca nas predicoes
    fora da dobra dela e grava a nova decisao em doencas_decisao.json (a decisao anterior fica dentro dela, em
    'anterior'). Depois: final e teste."""
    arq_res, arq_oof = conj.pasta / "doencas_cv_por_doenca.json", conj.pasta / "doencas_cv_por_doenca_oof.csv"
    arq_dec = conj.pasta / "doencas_decisao.json"
    if not arq_res.exists() or not arq_oof.exists() or not arq_dec.exists():
        raise SystemExit("rode hough_doencas.py cv e comparar antes")
    with open(arq_res, encoding="utf-8") as f:
        res = json.load(f)
    with open(arq_dec, encoding="utf-8") as f:
        anterior = json.load(f)
    if anterior.get("nivel1_por_doenca"):
        raise SystemExit("o nivel 1 por doenca ja foi adotado (doencas_decisao.json)")
    if not res["substitui_o_atual"]:
        raise SystemExit("a alternativa nao passou a regra em comparar: nada a adotar")
    tipo = res["escolhida"].removeprefix("por_doenca_")
    y, p = _yp(pd.read_csv(arq_oof, dtype={"id": str}), tipo)
    limiares = escolher_limiares(y, p)
    por_doenca, resumo = metricas(y, p, limiares)
    dec = {"versao": "v2", "tipo": tipo, "nivel1_por_doenca": True, "limiares": dict(zip(DOENCAS, map(float, limiares))),
           "variante": anterior["variante"], "f1_macro_por_dobra": res["f1_macro_por_dobra"], "fora_da_dobra": resumo,
           "por_doenca": por_doenca.round(4).to_dict("records"),
           "regra": "adotada por hough_doencas.py comparar: " + res["regra"], "anterior": anterior}
    with open(arq_dec, "w", encoding="utf-8") as f:
        json.dump(dec, f, indent=2, ensure_ascii=False)
    print(f"adotado: nivel 1 por doenca + {tipo} | limiares {dec['limiares']}")
    print(por_doenca.round(3).to_string(index=False))
    print(f"fora da dobra: F1 macro {resumo['f1_macro']:.3f}, acerto exato {resumo['acerto_exato']:.3f}")
    return dec


def ajustar_final(conj, hough=None, dados=None, artefato=ARTEFATO):
    """Ajusta o classificador escolhido em treino + validacao (folhas com doenca; pontuacoes dos circulos fora da
    dobra, como no treino do nivel 2) e grava o modelo. Com o nivel 1 por doenca adotado (adotar), ajusta tambem esse
    nivel 1 e o guarda no modelo, para o aplicativo e o teste pontuarem os circulos do mesmo jeito."""
    arq = conj.pasta / "doencas_decisao.json"
    if not arq.exists():
        raise SystemExit(f"{arq} nao existe: rode hough_doencas.py cv")
    with open(arq, encoding="utf-8") as f:
        dec = json.load(f)
    if hough is None:
        hough, dados = _contexto(conj)
    tv = conj.meta[conj.meta["split"].isin(["train", "val"])].set_index("id")
    _, p_tv = ht.ajustar_nivel1(tv.index, tv["classe"], tv["grupo"], dados, conj.classes)
    tv_d = com_doenca(tv)
    x = ht.montar_x(list(tv_d.index), dados, p_tv, dec["variante"])
    n1_5 = None
    if dec.get("nivel1_por_doenca"):
        n1_5, p5_tv = nivel1_por_doenca(tv, dados)
        x = np.hstack([x, x_por_doenca(list(tv_d.index), dados, p5_tv)])
    modelo = {"doencas": DOENCAS, "modelos": ajustar_doencas(x, tv_d[DOENCAS].to_numpy(int), dec["tipo"]),
              "limiares": [float(dec["limiares"][d]) for d in DOENCAS], "tipo": dec["tipo"],
              "nivel1_doencas": n1_5, "versao": dec.get("versao", "v1"),
              "variante": dec["variante"], "param1": hough["param1"], "param2": hough["param2"],
              "conjunto": conj.nome, "metadados_sha256": conj.sha256()}
    artefato.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, artefato, compress=3)
    print(f"modelo das doencas {modelo['versao']} ({dec['tipo']}" + (", com nivel 1 por doenca" if n1_5 is not None else "")
          + f", {len(tv_d)} folhas com doenca de treino + validacao): {artefato}")
    return modelo


def _bootstrap(y, p, limiares, grupos, n=N_BOOTSTRAP, seed=SEED):
    """IC 95% do F1 macro e do acerto exato, sorteando grupos (as quase-copias ficam juntas)."""
    rng = np.random.default_rng(seed)
    unicos = np.unique(grupos)
    idx = {g: np.flatnonzero(grupos == g) for g in unicos}
    f1, ex = [], []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(unicos, len(unicos))])
        r = metricas(y[s], p[s], limiares)[1]
        f1.append(r["f1_macro"])
        ex.append(r["acerto_exato"])
    return {"f1_macro": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))],
            "acerto_exato": [float(np.percentile(ex, 2.5)), float(np.percentile(ex, 97.5))]}


def avaliar_teste(conj, hough=None, dados=None, artefato=ARTEFATO, n_bootstrap=N_BOOTSTRAP):
    """Pontua as folhas com doenca do teste UMA vez, com o modelo gravado e o nivel 1 do modelo de Hough (o mesmo
    caminho do aplicativo). Para se doencas_teste.json ja existe. Grava doencas_teste.json, doencas_teste_por_doenca.csv
    e doencas_predicoes_teste.csv."""
    if not artefato.exists():
        raise SystemExit(f"{artefato} nao existe: rode hough_doencas.py final")
    modelo = joblib.load(artefato)
    saida = conj.pasta / "doencas_teste.json"
    if saida.exists():
        raise SystemExit(f"{saida} ja existe: o teste e pontuado uma vez so")
    if modelo["metadados_sha256"] != conj.sha256():
        raise SystemExit("a divisao (metadados) mudou desde o treino do modelo das doencas")
    if hough is None:
        hough, dados = _contexto(conj)
    te = com_doenca(conj.meta[conj.meta["split"] == "test"]).set_index("id")
    ids = list(te.index)
    x = ht.montar_x(ids, dados, ht._pontuar(hough["nivel1"], ids, dados), modelo["variante"])
    if modelo.get("nivel1_doencas") is not None:
        x = np.hstack([x, x_por_doenca(ids, dados, pontuar5(modelo["nivel1_doencas"], ids, dados))])
    p = prever_doencas(modelo["modelos"], x)
    y = te[DOENCAS].to_numpy(int)
    por_doenca, resumo = metricas(y, p, modelo["limiares"])
    resumo["ic95"] = _bootstrap(y, p, np.asarray(modelo["limiares"]), te["grupo"].to_numpy(), n_bootstrap)
    conj.pasta.mkdir(parents=True, exist_ok=True)
    por_doenca.to_csv(conj.pasta / "doencas_teste_por_doenca.csv", index=False)
    pd.DataFrame({"id": ids, **{f"y_{d}": y[:, j] for j, d in enumerate(DOENCAS)},
                  **{f"p_{d}": p[:, j] for j, d in enumerate(DOENCAS)}}).to_csv(
        conj.pasta / "doencas_predicoes_teste.csv", index=False)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump({**resumo, "versao": modelo.get("versao", "v1"), "tipo": modelo["tipo"],
                   "nivel1_por_doenca": modelo.get("nivel1_doencas") is not None,
                   "limiares": dict(zip(DOENCAS, modelo["limiares"]))}, f, indent=2)
    print(por_doenca.round(3).to_string(index=False))
    print(f"teste ({resumo['n']} folhas com doenca): F1 macro {resumo['f1_macro']:.3f} (IC 95% "
          f"{resumo['ic95']['f1_macro'][0]:.3f} a {resumo['ic95']['f1_macro'][1]:.3f}), acerto exato "
          f"{resumo['acerto_exato']:.3f}")
    return resumo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("etapa", choices=["cv", "comparar", "adotar", "final", "teste"])
    args = ap.parse_args(argv)
    conj = carregar_conjunto("bracol")
    {"cv": rodar_cv, "comparar": comparar, "adotar": adotar, "final": ajustar_final,
     "teste": avaliar_teste}[args.etapa](conj)


if __name__ == "__main__":
    main()
