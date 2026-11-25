"""Passo 2 do aplicativo, quais doencas: uma probabilidade por doenca do BRACOL, para a folha com doenca.

Uma folha pode ter mais de uma doenca (os rotulos do BRACOL sao 0/1 por doenca), entao ha um classificador por doenca
(multirrotulo), cada um com o seu limiar; nenhuma doenca vale mais que outra. O modelo:
- arvores: um gradient boosting por doenca, sobre os atributos do modelo de Hough (hough_treinar.montar_x, com as
  pontuacoes dos circulos pelo nivel 1 dele) e o resumo de um nivel 1 por doenca (5 classes: saudavel e as quatro
  doencas, treinado so com folhas de uma doenca);
- a probabilidade final de cada doenca e a media com a ConvNeXt-Tiny (convnext_bracol.py), que ve a foto inteira:
  PESO_ARVORES nas arvores e o resto na rede. Sem a rede, valem as arvores, com os limiares delas.
Essa configuracao foi escolhida antes, na validacao cruzada (docs/RELATORIO_FINAL.md); aqui so ela e treinada. Treino
e avaliacao usam so as folhas com pelo menos uma doenca.

Uso (na raiz do repositorio, com PYTHONPATH=src). Precisa do modelo de Hough do BRACOL (models/bracol_hough.joblib),
do cache de atributos (hough_treinar.py extrair --bracol) e das previsoes fora da dobra da rede (convnext_bracol.py cv):
  python -P src/hough_doencas.py cv     # fora da dobra (as 3 dobras do modelo de Hough): o limiar de cada doenca
  python -P src/hough_doencas.py final  # ajusta em treino + validacao e grava models/bracol_doencas.joblib
  python -P src/hough_doencas.py teste  # pontua o teste UMA vez (para se doencas_teste.json ja existe)

Resultados em outputs/hough_bracol/ (doencas_*).
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

import hough as rh
import hough_treinar as ht
from comum import MODELOS, SEED
from hough_dados import DOENCAS_BRACOL, carregar_conjunto

DOENCAS = list(DOENCAS_BRACOL)  # miner, rust, phoma, cercospora
PESO_ARVORES = 0.25  # peso das arvores na media com a ConvNeXt-Tiny (0,75 na rede)
ARTEFATO = MODELOS / "bracol_doencas.joblib"
N_BOOTSTRAP = 1000


def _esquema(modelo):
    """(classes, classe positiva) do modelo de Hough em que os atributos se apoiam; sem as chaves, as do padrao
    (hough_treinar.CLASSES e POSITIVA: saudavel ou doenca)."""
    return list(modelo.get("classes") or ht.CLASSES), modelo.get("positiva", ht.POSITIVA)


# ---------------------------------------------------------------- modelo
def com_doenca(meta):
    """As folhas de meta (colunas DOENCAS) com pelo menos uma doenca."""
    return meta[meta[DOENCAS].sum(axis=1) > 0]


def ajustar_doencas(x, y, fabrica=ht.classificador):
    """Um classificador binario por doenca (gradient boosting; os testes trocam, com poucas folhas). x: (n, k);
    y: (n, len(DOENCAS)) 0/1. Doenca sem exemplos positivos e negativos no treino: um numero (a fracao de positivos),
    usado como probabilidade."""
    y = np.asarray(y, int)
    return [float(y[:, j].mean()) if len(np.unique(y[:, j])) < 2 else fabrica().fit(x, y[:, j])
            for j in range(y.shape[1])]


def prever_doencas(modelos, x):
    """(n, len(modelos)): probabilidade de cada doenca."""
    p = np.zeros((len(x), len(modelos)))
    for j, m in enumerate(modelos):
        p[:, j] = m if isinstance(m, float) else m.predict_proba(x)[:, list(m.classes_).index(1)]
    return p


def montar_x(ids, dados, p_circ, p5, classes, positiva):
    """Atributos das arvores: os do modelo de Hough (pontuacoes dos circulos pelo nivel 1 dele) e o resumo do nivel 1
    por doenca (p5: id -> (N, 5))."""
    return np.hstack([ht.montar_x(ids, dados, p_circ, classes, positiva), x_por_doenca(ids, dados, p5)])


def prever_foto(modelo, dados_foto, p_circ):
    """Uma foto ({"circulos", "at_circulos", "at_folha"}) e as pontuacoes (N, k) dos seus circulos pelo nivel 1 do
    modelo de Hough -> {doenca: probabilidade das arvores}. Usado pelo aplicativo (hough_inferir.py)."""
    dados = {"foto": dados_foto}
    p5 = pontuar5(modelo["nivel1_doencas"], ["foto"], dados)
    x = montar_x(["foto"], dados, {"foto": np.asarray(p_circ)}, p5, modelo["classes_hough"], modelo["positiva_hough"])
    return {d: float(v) for d, v in zip(modelo["doencas"], prever_doencas(modelo["modelos"], x)[0])}


def media(p_arvores, p_convnext, w=PESO_ARVORES):
    """A probabilidade final de cada doenca: w * arvores + (1 - w) * ConvNeXt-Tiny."""
    return w * np.asarray(p_arvores, float) + (1.0 - w) * np.asarray(p_convnext, float)


def combinar(modelo, p_arvores, p_convnext=None):
    """As probabilidades finais e os limiares que valem para elas: com a rede, a media e os limiares da media; sem a
    rede, as arvores, com os limiares delas. p_arvores, p_convnext: (n, 4) ou (4,). Devolve (p, limiares, usou_rede)."""
    if p_convnext is None:
        return np.asarray(p_arvores, float), np.asarray(modelo["limiares"], float), False
    f = modelo["fusao"]
    return media(p_arvores, p_convnext, f["w"]), np.asarray(f["limiares"], float), True


# ---------------------------------------------------------------- nivel 1 por doenca
# O nivel 1 do modelo de Hough so conhece saudavel / doenca. Aqui, um nivel 1 com as 5 classes (saudavel e as 4
# doencas), treinado so com circulos de folhas de rotulo unico (onde o rotulo fraco "o circulo tem a doenca da folha"
# vale), e um resumo por classe: pontuacoes e como sao os 5 circulos mais parecidos com cada uma.
CLASSES5 = ["healthy"] + DOENCAS
ATRIBUTOS_TOP5 = ["raio", "apoio", "croma", "mapa_media", "delta_L", "delta_a", "delta_b", "centro_L", "variancia",
                  "blob_area", "blob_circularidade", "blob_alongamento"]
_IDX_TOP5 = [rh.NOMES_CIRCULO.index(n) for n in ATRIBUTOS_TOP5]
NOMES_POR_DOENCA = ["n_circulos"] + [f"{c}_{e}" for c in CLASSES5 for e in
                                     ["max", "top3", "top5", "media", "soma", "n_05", "frac_maior", "area"]
                                     + [f"top5_{a}" for a in ATRIBUTOS_TOP5]]


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
        p.update(pontuar5(ht.classificador().fit(*_circulos_rotulados(ids[tr], rot, dados)), ids[va], dados))
    return ht.classificador().fit(*_circulos_rotulados(ids, rot, dados)), p


def estatisticas_por_doenca(p, at_circulos, raios, area_folha=rh.LADO_FOLHA ** 2):
    """Resumo dos circulos de uma foto, classe a classe (ordem de NOMES_POR_DOENCA): quantos circulos; para cada classe
    de CLASSES5, maximo, media dos 3 e dos 5 maiores, media e soma de P(classe), quantos passam de 0,5, fracao dos
    circulos em que ela e a mais provavel, area pesada por P(classe) e a media de ATRIBUTOS_TOP5 nos 5 circulos mais
    parecidos com ela. Sem circulos: zeros."""
    p = np.asarray(p, np.float64).reshape(-1, len(CLASSES5))
    if not len(p):
        return np.zeros(len(NOMES_POR_DOENCA), np.float32)
    at = np.asarray(at_circulos, np.float64)[:, _IDX_TOP5]
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


# ---------------------------------------------------------------- metricas
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


def f1_macro_por_dobra(y, p, dobra):
    """F1 macro de cada dobra, com os limiares escolhidos nas OUTRAS dobras (o numero honesto)."""
    return [metricas(y[dobra == k], p[dobra == k], escolher_limiares(y[dobra != k], p[dobra != k]))[1]["f1_macro"]
            for k in sorted(np.unique(dobra))]


# ---------------------------------------------------------------- validacao cruzada
def oof_doencas(meta, dados, classes, positiva=ht.POSITIVA, fabrica=ht.classificador):
    """Probabilidades das arvores fora da dobra para as folhas com doenca do treino + validacao: DataFrame id, dobra,
    grupo, y_<doenca>, p_arvores_<doenca>. As dobras sao as do modelo de Hough (hough_treinar.dobras). Os niveis 1 de
    cada dobra (o do modelo de Hough e o por doenca) sao ajustados nas folhas de treino da dobra, saudaveis tambem; as
    folhas de treino sao pontuadas fora da dobra, e as de fora, pelos niveis 1 da dobra."""
    d = ht.dobras(meta)
    base = meta[meta["split"].isin(["train", "val"])].set_index("id").loc[d.index]
    partes = []
    for k in range(ht.N_DOBRAS):
        tr, fora = base[(d != k).to_numpy()], base[(d == k).to_numpy()]
        tr_d, fora_d = com_doenca(tr), com_doenca(fora)
        ids_tr, ids_fora = list(tr_d.index), list(fora_d.index)
        n1, p_tr = ht.ajustar_nivel1(tr.index, tr["classe"], tr["grupo"], dados, classes)
        n1_5, p5_tr = nivel1_por_doenca(tr, dados)
        x_tr = montar_x(ids_tr, dados, p_tr, p5_tr, classes, positiva)
        x_fora = montar_x(ids_fora, dados, ht._pontuar(n1, ids_fora, dados, len(classes)),
                          pontuar5(n1_5, ids_fora, dados), classes, positiva)
        p = prever_doencas(ajustar_doencas(x_tr, tr_d[DOENCAS].to_numpy(int), fabrica), x_fora)
        partes.append(pd.DataFrame({"id": fora_d.index.to_numpy(), "dobra": k, "grupo": fora_d["grupo"].to_numpy(),
                                    **{f"y_{dz}": fora_d[dz].to_numpy(int) for dz in DOENCAS},
                                    **{f"p_arvores_{dz}": p[:, j] for j, dz in enumerate(DOENCAS)}}))
    return pd.concat(partes, ignore_index=True)


def _contexto(conj):
    """Modelo de Hough gravado e os atributos do cache do conjunto."""
    if not conj.artefato.exists():
        raise SystemExit(f"{conj.artefato} nao existe: rode hough_treinar.py final --dataset {conj.nome}")
    return joblib.load(conj.artefato), ht.atributos_do_conjunto(conj)


def rodar_cv(conj, hough=None, dados=None, cnn=None, fabrica=ht.classificador):
    """Previsoes fora da dobra das arvores e da media com a ConvNeXt-Tiny, e o limiar de cada doenca (o de maior F1
    fora da dobra), para as duas. cnn: id, p_<doenca> da rede fora da dobra (padrao: convnext_oof.csv, de
    convnext_bracol.py cv). Grava doencas_cv_oof.csv e doencas_decisao.json em conj.pasta."""
    if hough is None:
        hough, dados = _contexto(conj)
    if cnn is None:
        arq_cnn = conj.pasta / "convnext_oof.csv"
        if not arq_cnn.exists():
            raise SystemExit(f"{arq_cnn} nao existe: rode convnext_bracol.py cv")
        cnn = pd.read_csv(arq_cnn, dtype={"id": str})
    classes, positiva = _esquema(hough)
    oof = oof_doencas(conj.meta, dados, classes, positiva, fabrica)
    cnn = cnn.astype({"id": str}).set_index("id")
    faltam = set(oof["id"].astype(str)) - set(cnn.index)
    if faltam:
        raise SystemExit(f"a ConvNeXt-Tiny nao tem predicao fora da dobra para {len(faltam)} folhas: rode "
                         "convnext_bracol.py cv")
    y = oof[[f"y_{dz}" for dz in DOENCAS]].to_numpy(int)
    p_arv = oof[[f"p_arvores_{dz}" for dz in DOENCAS]].to_numpy()
    p_cnn = cnn.loc[oof["id"].astype(str), [f"p_{dz}" for dz in DOENCAS]].to_numpy()
    p = media(p_arv, p_cnn)
    oof = oof.assign(**{f"p_convnext_{dz}": p_cnn[:, j] for j, dz in enumerate(DOENCAS)},
                     **{f"p_{dz}": p[:, j] for j, dz in enumerate(DOENCAS)})
    limiares_arv, limiares = escolher_limiares(y, p_arv), escolher_limiares(y, p)
    dobra = oof["dobra"].to_numpy()
    resumo = {nome: {**metricas(y, pp, lim)[1], "f1_macro_por_dobra": [round(v, 4) for v in f1_macro_por_dobra(y, pp, dobra)]}
              for nome, pp, lim in (("arvores", p_arv, limiares_arv), ("media", p, limiares))}
    por_doenca = metricas(y, p, limiares)[0]
    dec = {"peso_arvores": PESO_ARVORES, "limiares": dict(zip(DOENCAS, map(float, limiares))),
           "limiares_arvores": dict(zip(DOENCAS, map(float, limiares_arv))), "fora_da_dobra": resumo,
           "por_doenca": por_doenca.round(4).to_dict("records")}
    conj.pasta.mkdir(parents=True, exist_ok=True)
    oof.to_csv(conj.pasta / "doencas_cv_oof.csv", index=False)
    with open(conj.pasta / "doencas_decisao.json", "w", encoding="utf-8") as f:
        json.dump(dec, f, indent=2, ensure_ascii=False)
    for nome in ("arvores", "media"):
        r = resumo[nome]
        print(f"{nome}: F1 macro fora da dobra por dobra {r['f1_macro_por_dobra']} (limiares das outras dobras) | "
              f"com os limiares escolhidos: F1 macro {r['f1_macro']:.3f}, acerto exato {r['acerto_exato']:.3f}")
    print(f"limiares da media (peso {PESO_ARVORES} nas arvores): {dec['limiares']}")
    print(por_doenca.round(3).to_string(index=False))
    return dec


# ---------------------------------------------------------------- modelo final e teste
def ajustar_final(conj, hough=None, dados=None, artefato=ARTEFATO, fabrica=ht.classificador):
    """Ajusta os niveis 1 e os classificadores das doencas em treino + validacao (folhas com doenca; pontuacoes dos
    circulos fora da dobra, como no treino do nivel 2) e grava o modelo, com os limiares de doencas_decisao.json."""
    arq = conj.pasta / "doencas_decisao.json"
    if not arq.exists():
        raise SystemExit(f"{arq} nao existe: rode hough_doencas.py cv")
    with open(arq, encoding="utf-8") as f:
        dec = json.load(f)
    if hough is None:
        hough, dados = _contexto(conj)
    classes, positiva = _esquema(hough)
    tv = conj.meta[conj.meta["split"].isin(["train", "val"])].set_index("id")
    _, p_tv = ht.ajustar_nivel1(tv.index, tv["classe"], tv["grupo"], dados, classes)
    n1_5, p5_tv = nivel1_por_doenca(tv, dados)
    tv_d = com_doenca(tv)
    x = montar_x(list(tv_d.index), dados, p_tv, p5_tv, classes, positiva)
    modelo = {"doencas": DOENCAS, "modelos": ajustar_doencas(x, tv_d[DOENCAS].to_numpy(int), fabrica),
              "nivel1_doencas": n1_5, "limiares": [float(dec["limiares_arvores"][d]) for d in DOENCAS],
              "fusao": {"w": float(dec["peso_arvores"]), "limiares": [float(dec["limiares"][d]) for d in DOENCAS]},
              "classes_hough": classes, "positiva_hough": positiva, "param1": hough["param1"],
              "param2": hough["param2"], "conjunto": conj.nome, "metadados_sha256": conj.sha256()}
    artefato.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, artefato, compress=3)
    print(f"modelo das doencas ({len(tv_d)} folhas com doenca de treino + validacao; media com a ConvNeXt-Tiny, peso "
          f"{modelo['fusao']['w']:.2f} nas arvores): {artefato}")
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


def _convnext_teste(meta):
    """Probabilidades da ConvNeXt-Tiny final para as folhas de meta (a mesma rede do aplicativo)."""
    import convnext_bracol as cb
    cb.preparar_cache(meta.reset_index())
    return cb.prever(cb.carregar(), meta.reset_index())


def avaliar_teste(conj, hough=None, dados=None, artefato=ARTEFATO, n_bootstrap=N_BOOTSTRAP, convnext=None):
    """Pontua as folhas com doenca do teste UMA vez, com o modelo gravado, o nivel 1 do modelo de Hough (o mesmo
    caminho do aplicativo) e a ConvNeXt-Tiny final. Para se doencas_teste.json ja existe. Grava doencas_teste.json,
    doencas_teste_por_doenca.csv e doencas_predicoes_teste.csv.
    convnext: funcao meta -> (n, 4) (padrao: a rede gravada em models/bracol_convnext.pt)."""
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
    classes, positiva = _esquema(hough)
    x = montar_x(ids, dados, ht._pontuar(hough["nivel1"], ids, dados, len(classes)),
                 pontuar5(modelo["nivel1_doencas"], ids, dados), classes, positiva)
    p_arv = prever_doencas(modelo["modelos"], x)
    p_cnn = (convnext or _convnext_teste)(te)
    p, limiares, _ = combinar(modelo, p_arv, p_cnn)
    y = te[DOENCAS].to_numpy(int)
    por_doenca, resumo = metricas(y, p, limiares)
    resumo["ic95"] = _bootstrap(y, p, limiares, te["grupo"].to_numpy(), n_bootstrap)
    resumo["arvores_sozinhas"] = metricas(y, p_arv, modelo["limiares"])[1]  # so para descrever (nada e escolhido)
    conj.pasta.mkdir(parents=True, exist_ok=True)
    por_doenca.to_csv(conj.pasta / "doencas_teste_por_doenca.csv", index=False)
    pd.DataFrame({"id": ids, **{f"y_{d}": y[:, j] for j, d in enumerate(DOENCAS)},
                  **{f"p_{d}": p[:, j] for j, d in enumerate(DOENCAS)},
                  **{f"p_arvores_{d}": p_arv[:, j] for j, d in enumerate(DOENCAS)},
                  **{f"p_convnext_{d}": p_cnn[:, j] for j, d in enumerate(DOENCAS)}}).to_csv(
        conj.pasta / "doencas_predicoes_teste.csv", index=False)
    with open(saida, "w", encoding="utf-8") as f:
        json.dump({**resumo, "fusao": modelo["fusao"], "limiares": dict(zip(DOENCAS, map(float, limiares)))}, f,
                  indent=2)
    print(por_doenca.round(3).to_string(index=False))
    print(f"teste ({resumo['n']} folhas com doenca): F1 macro {resumo['f1_macro']:.3f} (IC 95% "
          f"{resumo['ic95']['f1_macro'][0]:.3f} a {resumo['ic95']['f1_macro'][1]:.3f}), acerto exato "
          f"{resumo['acerto_exato']:.3f}")
    return resumo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("etapa", choices=["cv", "final", "teste"])
    args = ap.parse_args(argv)
    conj = carregar_conjunto("bracol")
    {"cv": rodar_cv, "final": ajustar_final, "teste": avaliar_teste}[args.etapa](conj)


if __name__ == "__main__":
    main()
