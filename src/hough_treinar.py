"""Passo 1 do aplicativo, saudavel ou doenca: o modelo de Hough de dois niveis.

Uso (na raiz do repositorio, com PYTHONPATH=src):
  python -P src/hough_treinar.py mascaras --bracol  # SAM em todas as folhas do BRACOL (GPU)
  python -P src/hough_treinar.py extrair --bracol   # circulos do Hough e atributos de todas as folhas (CPU)
  python -P src/hough_treinar.py cv --bracol        # estimativa fora da dobra (3 dobras do treino + validacao)
  python -P src/hough_treinar.py final --bracol     # ajusta em treino + validacao e grava models/bracol_hough.joblib

O modelo tem dois niveis, os dois com gradient boosting:
- nivel 1, por circulo: a probabilidade de cada circulo do Hough ser de folha saudavel ou com doenca (cada circulo
  herda a classe da sua folha);
- nivel 2, por folha: o resumo das pontuacoes dos circulos, com os atributos da folha inteira, da P(saudavel) e
  P(doenca). A folha e doenca se tem qualquer uma das quatro doencas; vale a classe mais provavel.
Essa configuracao foi escolhida antes, na validacao cruzada (docs/RELATORIO_FINAL.md); aqui so ela e treinada. O teste
so e lido por hough_avaliar.py.
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

import hough as rh
from comum import SEED
from hough_dados import CLASSES_BRACOL, adicionar_opcao_dataset, carregar_conjunto


# ---------------------------------------------------------------- extracao em lote
def carregar_mascara(id_, forma, pastas):
    """Mascara da folha em cache (<pasta>/<id>.png, metade da resolucao), ampliada para forma (A, L). None se nao ha."""
    for pasta in pastas:
        arq = Path(pasta) / f"{id_}.png"
        if arq.exists():
            m = cv2.imread(str(arq), cv2.IMREAD_GRAYSCALE)
            return cv2.resize(m, (forma[1], forma[0]), interpolation=cv2.INTER_NEAREST) > 0
    return None


def confiancas_sam(arq):
    """id -> confianca da mascara do SAM (csv com id e confianca); fotos sem mascara valida ficam de fora."""
    if not Path(arq).exists():
        return {}
    t = pd.read_csv(arq).dropna(subset=["confianca"])
    return dict(zip(t["id"].astype(str), t["confianca"].astype(float)))


def _um_fio():
    cv2.setNumThreads(1)  # um processo por nucleo


def _ler_foto(arquivo):
    """Foto BGR lida do arquivo."""
    img = cv2.imread(str(arquivo), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"nao consegui ler a foto: {arquivo}")
    return img


def _extrair_uma(tarefa):
    id_, arquivo, pastas, confianca = tarefa
    img = _ler_foto(arquivo)
    m = carregar_mascara(id_, img.shape[:2], pastas)
    r = rh.extrair(img, m, confianca if m is not None else 0.0)
    return id_, {"circulos": r["circulos"], "at_circulos": r["at_circulos"], "at_folha": r["at_folha"]}


def extrair_tudo(meta, pastas=None, confiancas=None, processos=None, cache=None):
    """Circulos e atributos de todas as fotos de meta (colunas id e arquivo, o caminho da foto no disco): id ->
    {"circulos", "at_circulos", "at_folha"}. Os atributos nao usam o rotulo. pastas: onde procurar as mascaras da
    folha (sem mascara, o contorno sai da cor). Com `cache` (arquivo .joblib), grava e reaproveita."""
    if cache is not None and Path(cache).exists():
        return joblib.load(cache)
    confiancas = {} if confiancas is None else confiancas
    pastas = [str(p) for p in (pastas or [])]
    tarefas = [(str(r.id), r.arquivo, pastas, float(confiancas.get(str(r.id), 0.0))) for r in meta.itertuples()]
    processos = processos or max(1, (os.cpu_count() or 2) - 1)
    if processos == 1:
        pares = [_extrair_uma(t) for t in tarefas]
    else:
        with Pool(processos, initializer=_um_fio) as pool:
            pares = pool.map(_extrair_uma, tarefas, chunksize=8)
    dados = dict(pares)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dados, cache, compress=3)
    return dados


def atributos_do_conjunto(conj):
    """Circulos e atributos de todas as fotos de um conjunto (hough_dados.Conjunto), do cache dele se ja foram
    extraidos."""
    return extrair_tudo(conj.meta, pastas=conj.pastas_mascaras, confiancas=confiancas_sam(conj.arquivo_confiancas),
                        cache=conj.cache(rh.PARAM1, rh.PARAM2))


def gerar_mascaras_do_conjunto(conj):
    """SAM (ponto no centro) em todas as fotos de um conjunto que nao tem mascaras prontas: grava a mascara de cada
    foto (metade da resolucao) e a confianca em conj.arquivo_confiancas. Retoma de onde parou."""
    import torch
    from PIL import Image

    from folha_sam import SegmentadorSAM
    pasta = Path(conj.pastas_mascaras[0])
    pasta.mkdir(parents=True, exist_ok=True)
    conj.pasta.mkdir(parents=True, exist_ok=True)
    linhas = []
    if Path(conj.arquivo_confiancas).exists():
        linhas = pd.read_csv(conj.arquivo_confiancas).astype({"id": str}).to_dict("records")
    prontas = {r["id"] for r in linhas}
    faltam = [r for r in conj.meta.itertuples() if str(r.id) not in prontas]
    print(f"{conj.nome}: {len(faltam)} fotos sem mascara; rodando o SAM...", flush=True)
    if not faltam:
        return
    sam = SegmentadorSAM(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for n, r in enumerate(faltam, 1):
        img = _ler_foto(r.arquivo)
        m, conf = rh.escolher_mascara_hough(*sam(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if m is not None:
            h, w = m.shape
            Image.fromarray(m.astype(np.uint8) * 255).resize((w // 2, h // 2), Image.NEAREST).save(pasta / f"{r.id}.png")
        linhas.append({"id": str(r.id), "ok": m is not None, "confianca": conf if m is not None else np.nan,
                       "area": float(m.mean()) if m is not None else np.nan})
        if n % 200 == 0 or n == len(faltam):
            pd.DataFrame(linhas).to_csv(conj.arquivo_confiancas, index=False)
            print(f"{n}/{len(faltam)} fotos", flush=True)
    tab = pd.DataFrame(linhas)
    print(f"mascaras gravadas: {int(tab['ok'].sum())} de {len(tab)} (as outras usam o contorno pela cor) | confianca "
          f"mediana {tab['confianca'].median():.3f}")


def figura_propostas(conj, caminho, n=12):
    """Fotos de TREINO sorteadas, com o contorno da folha e os circulos do Hough (vermelho = os 15 de mais votos),
    ao lado do mapa de lesao. So para conferir a olho."""
    conf = confiancas_sam(conj.arquivo_confiancas)
    meta = conj.meta
    amostra = meta[meta["split"] == "train"].groupby("classe", group_keys=False).sample(n // 3, random_state=SEED)
    tiles = []
    for r in amostra.itertuples():
        img = _ler_foto(r.arquivo)
        m = carregar_mascara(r.id, img.shape[:2], conj.pastas_mascaras)
        e = rh.extrair(img, m, conf.get(str(r.id), 0.0) if m is not None else 0.0)
        vis = e["folha"].img.copy()
        cont, _ = cv2.findContours(e["folha"].mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cont, -1, (255, 0, 255), 2)
        for i, (x, y, raio) in enumerate(e["circulos"]):
            cv2.circle(vis, (int(x), int(y)), int(raio), (0, 0, 255) if i < 15 else (0, 255, 255), 2)
        par = np.hstack([vis, cv2.cvtColor(e["mapa"], cv2.COLOR_GRAY2BGR)])
        par = cv2.resize(par, (900, max(1, int(par.shape[0] * 900 / par.shape[1]))))
        par = cv2.copyMakeBorder(par, 0, max(0, 400 - par.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:400]
        cv2.putText(par, f"{r.id} {r.classe} n={len(e['circulos'])}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)
        tiles.append(par)
    linhas = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
    cv2.imwrite(str(caminho), np.vstack(linhas), [cv2.IMWRITE_JPEG_QUALITY, 85])


def resumo_propostas(meta, dados):
    """Circulos por classe no TREINO: media, mediana e fracao das fotos com pelo menos 1 circulo."""
    tr = meta[meta["split"] == "train"]
    n = tr["id"].map(lambda i: len(dados[i]["circulos"]))
    return tr.assign(n=n).groupby("classe")["n"].agg(media="mean", mediana="median",
                                                      com_circulo=lambda s: (s > 0).mean())


# ---------------------------------------------------------------- modelo de dois niveis
CLASSES = CLASSES_BRACOL  # saudavel ou doenca: o padrao das funcoes abaixo
POSITIVA = "disease"  # a classe acompanhada pelos resumos dos circulos
_I_POS = CLASSES.index(POSITIVA)
LIMIAR = 0.5  # P(doenca) >= 0,5: a classe mais provavel das duas
N_DOBRAS, N_INTERNAS = 3, 3
GRADE_LIMIAR = np.round(np.arange(0.20, 0.801, 0.02), 2)
_CURTO = {"disease": "doenca"}  # nome curto da classe nos nomes dos atributos


def _esquema(classes=None, positiva=POSITIVA):
    """(lista de classes, indice da classe positiva). Padrao: CLASSES e POSITIVA."""
    classes = list(CLASSES if classes is None else classes)
    return classes, classes.index(positiva)


def nomes_saco(classes=None, positiva=POSITIVA):
    """Nomes do resumo dos circulos (estatisticas_saco) para as classes de um conjunto."""
    classes, _ = _esquema(classes, positiva)
    pos = _CURTO.get(positiva, positiva)
    return (["n_circulos"] + [f"{e}_{_CURTO.get(c, c)}" for c in classes for e in ("max", "top3", "top5", "media", "soma")]
            + [f"n_{pos}_05", f"n_{pos}_07", f"area_{pos}"])


NOMES_SACO = nomes_saco()  # os do padrao (saudavel ou doenca)


def estatisticas_saco(p, raios, area_folha=rh.LADO_FOLHA ** 2, n_classes=len(CLASSES), i_pos=_I_POS):
    """Resume as pontuacoes dos circulos de uma foto (p: (N, n_classes), nivel 1) num vetor fixo, na ordem de
    nomes_saco: quantos circulos; por classe, maximo, media dos 3 e dos 5 maiores, media e soma; quantos parecem da
    classe positiva i_pos (> 0,5 e > 0,7); e a area dos circulos pesada pela pontuacao dela, em fracao da folha.
    Sem circulos: zeros."""
    p = np.asarray(p, np.float64).reshape(-1, n_classes)
    if not len(p):
        return np.zeros(1 + 5 * n_classes + 3, np.float32)
    saida = [len(p)]
    for c in range(n_classes):
        ordenado = np.sort(p[:, c])[::-1]
        saida += [ordenado[0], ordenado[:3].mean(), ordenado[:5].mean(), ordenado.mean(), ordenado.sum()]
    r = np.asarray(raios, np.float64)
    saida += [(p[:, i_pos] > 0.5).sum(), (p[:, i_pos] > 0.7).sum(), (p[:, i_pos] * np.pi * r ** 2).sum() / area_folha]
    return np.asarray(saida, np.float32)


def montar_x(ids, dados, p_circ, classes=None, positiva=POSITIVA):
    """Matriz do nivel 2: o resumo das pontuacoes dos circulos (estatisticas_saco) e os atributos da folha inteira.
    p_circ: id -> (N, len(classes)) com a pontuacao dos circulos. classes e positiva: as do conjunto (padrao: CLASSES
    e POSITIVA)."""
    classes, i_pos = _esquema(classes, positiva)
    saco = np.stack([estatisticas_saco(p_circ[i], dados[i]["circulos"][:, 2], n_classes=len(classes), i_pos=i_pos)
                     for i in ids])
    return np.hstack([saco, np.stack([dados[i]["at_folha"] for i in ids])]).astype(np.float32)


def classificador():
    """O classificador dos dois niveis (e das doencas, em hough_doencas.py): gradient boosting com classes
    balanceadas."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)


def _circulos_de(ids, y, dados):
    """Empilha os circulos das fotos: atributos e o rotulo da FOTO de cada um (rotulo fraco)."""
    x = np.vstack([dados[i]["at_circulos"] for i in ids])
    yc = np.concatenate([np.full(len(dados[i]["at_circulos"]), c) for i, c in zip(ids, y)])
    return x, yc


def _pontuar(modelo, ids, dados, n_classes=len(CLASSES)):
    """id -> (N, n_classes): probabilidade de cada circulo ser de foto de cada classe."""
    saida = {}
    for i in ids:
        at = dados[i]["at_circulos"]
        p = np.zeros((len(at), n_classes))
        if len(at):
            p[:, modelo.classes_] = modelo.predict_proba(at)
        saida[i] = p
    return saida


def pontuar_fora_da_dobra(ids, y, grupos, dados, fabrica=classificador, n_dobras=N_INTERNAS, n_classes=len(CLASSES)):
    """Pontua os circulos de cada foto com um nivel 1 que nao viu nenhuma foto do mesmo grupo."""
    from sklearn.model_selection import StratifiedGroupKFold
    ids, y, grupos = np.asarray(ids), np.asarray(y), np.asarray(grupos)
    saida = {}
    sgkf = StratifiedGroupKFold(n_splits=n_dobras, shuffle=True, random_state=SEED)
    for tr, va in sgkf.split(ids, y, grupos):
        x, yc = _circulos_de(ids[tr], y[tr], dados)
        saida.update(_pontuar(fabrica().fit(x, yc), ids[va], dados, n_classes))
    return saida


def _indices(classes, nomes=None):
    """Indice de cada classe em `nomes` (os nomes das classes do conjunto; padrao: CLASSES). Inteiros passam
    direto. Nome desconhecido: ValueError."""
    nomes = list(CLASSES if nomes is None else nomes)
    return np.array([nomes.index(c) if isinstance(c, str) else int(c) for c in classes])


def ajustar_nivel1(ids, classes, grupos, dados, nomes=None):
    """Nivel 1 ajustado em todas as fotos de ids, e a pontuacao FORA DA DOBRA (por grupo) dos circulos dessas
    mesmas fotos, que e o que o nivel 2 pode ver no treino."""
    ids, y = list(ids), _indices(classes, nomes)
    k = len(CLASSES if nomes is None else nomes)
    return classificador().fit(*_circulos_de(ids, y, dados)), pontuar_fora_da_dobra(ids, y, grupos, dados, n_classes=k)


def ajustar(ids, classes, grupos, dados, nomes=None, positiva=POSITIVA, fabrica=classificador):
    """Treina os dois niveis. O nivel 2 so ve pontuacoes de circulos feitas fora da dobra, por grupo.
    classes: nomes (os de `nomes`; padrao CLASSES) ou indices. fabrica: o classificador do nivel 2 (os testes trocam,
    com poucas fotos). Devolve {"nivel1", "nivel2", "classes", "positiva"}."""
    ids, y = list(ids), _indices(classes, nomes)
    nomes = list(CLASSES if nomes is None else nomes)
    nivel1, p_circ = ajustar_nivel1(ids, y, grupos, dados, nomes)
    nivel2 = fabrica().fit(montar_x(ids, dados, p_circ, nomes, positiva), y)
    return {"nivel1": nivel1, "nivel2": nivel2, "classes": nomes, "positiva": positiva}


def prever(modelo, ids, dados):
    """Probabilidades (n, k), na ordem das classes do modelo (modelo["classes"]; sem a chave, CLASSES)."""
    ids = list(ids)
    classes = list(modelo.get("classes") or CLASSES)
    positiva = modelo.get("positiva", POSITIVA)
    p_circ = _pontuar(modelo["nivel1"], ids, dados, len(classes))
    n2 = modelo["nivel2"]
    p = np.zeros((len(ids), len(classes)))
    p[:, n2.classes_] = n2.predict_proba(montar_x(ids, dados, p_circ, classes, positiva))
    return p


# ---------------------------------------------------------------- validacao cruzada e modelo final
def dobras(meta, n_dobras=N_DOBRAS, seed=SEED):
    """Dobra (0..n-1) de cada foto de treino + validacao, indexada pelo id. Um grupo nunca fica em duas dobras; as
    dobras sao estratificadas pela coluna estrato (a combinacao das doencas) quando ela existe, senao pela classe."""
    import warnings

    from sklearn.model_selection import StratifiedGroupKFold
    base = meta[meta["split"].isin(["train", "val"])].reset_index(drop=True)
    d = pd.Series(-1, index=base["id"].to_numpy(), name="dobra")
    estrato = base["estrato"] if "estrato" in base else base["classe"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sgkf = StratifiedGroupKFold(n_splits=n_dobras, shuffle=True, random_state=seed)
        for k, (_, fora) in enumerate(sgkf.split(base, estrato, base["grupo"])):
            d.iloc[fora] = k
    return d


def cv_externa(meta, dados, nomes=None, positiva=POSITIVA, fabrica=classificador):
    """Predicoes fora da dobra de todas as fotos de treino + validacao: id, dobra, classe, grupo, p_<classe>. Cada
    foto e prevista por um modelo treinado sem a dobra dela."""
    nomes = list(CLASSES if nomes is None else nomes)
    d = dobras(meta)
    base = meta[meta["split"].isin(["train", "val"])].set_index("id").loc[d.index]
    partes = []
    for k in range(N_DOBRAS):
        tr, fora = base[(d != k).to_numpy()], base[(d == k).to_numpy()]
        modelo = ajustar(tr.index, tr["classe"], tr["grupo"], dados, nomes, positiva, fabrica)
        p = prever(modelo, fora.index, dados)
        partes.append(pd.DataFrame({"id": fora.index, "dobra": k, "classe": fora["classe"].to_numpy(),
                                    "grupo": fora["grupo"].to_numpy(),
                                    **{f"p_{c}": p[:, j] for j, c in enumerate(nomes)}}))
    return pd.concat(partes, ignore_index=True)


def _f1(y, pred):
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def escolher_limiar(y_pos, p_pos, grade=GRADE_LIMIAR):
    """Limiar de P(classe positiva) com o maior F1; no empate, o mais perto de 0,5. Sem nenhum acerto possivel: 0,5.
    Usado pelo modelo das doencas (hough_doencas.py)."""
    y, p = np.asarray(y_pos, bool), np.asarray(p_pos, float)
    f1 = np.array([_f1(y, p >= t) for t in grade])
    if f1.max() <= 0:
        return 0.5
    melhores = grade[f1 >= f1.max() - 1e-12]
    return float(melhores[np.abs(melhores - 0.5).argmin()])


def f1_macro_por_dobra(oof):
    """F1 macro das classes em cada dobra, com a classe mais provavel: e o que o aplicativo faz no passo 1."""
    from sklearn.metrics import f1_score
    nomes = [c[2:] for c in oof.columns if c.startswith("p_")]
    pred = np.array(nomes)[oof[[f"p_{c}" for c in nomes]].to_numpy().argmax(1)]
    d = oof["dobra"].to_numpy()
    return pd.Series({int(k): float(f1_score(oof["classe"].to_numpy()[d == k], pred[d == k], labels=nomes,
                                             average="macro", zero_division=0)) for k in sorted(np.unique(d))},
                     name="f1_macro")


def rodar_cv(conj):
    """Estimativa fora da dobra do modelo no treino + validacao: grava cv_oof.csv na pasta do conjunto e mostra o F1
    macro (saudavel x doenca) de cada dobra. Nada e escolhido aqui; hough_avaliar.py mostra estes numeros ao lado dos
    do teste."""
    oof = cv_externa(conj.meta, atributos_do_conjunto(conj), conj.classes, conj.positiva)
    conj.pasta.mkdir(parents=True, exist_ok=True)
    oof.to_csv(conj.pasta / "cv_oof.csv", index=False)
    f1 = f1_macro_por_dobra(oof)
    print(f"{conj.nome}, fora da dobra ({len(oof)} fotos de treino + validacao): F1 macro (saudavel x doenca) por "
          f"dobra {[round(v, 3) for v in f1]} | media {f1.mean():.4f}")
    print(f"gravado: {conj.pasta / 'cv_oof.csv'}")
    return oof


def ajustar_final(conj):
    """Ajusta o modelo em treino + validacao do conjunto e grava conj.artefato."""
    meta = conj.meta
    tv = meta[meta["split"].isin(["train", "val"])]
    modelo = ajustar(tv["id"], tv["classe"], tv["grupo"], atributos_do_conjunto(conj), conj.classes, conj.positiva)
    modelo.update({"param1": rh.PARAM1, "param2": rh.PARAM2, "limiar": LIMIAR, "conjunto": conj.nome,
                   "nomes_circulo": rh.NOMES_CIRCULO, "nomes_folha": rh.NOMES_FOLHA,
                   "nomes_saco": nomes_saco(conj.classes, conj.positiva), "metadados_sha256": conj.sha256()})
    conj.artefato.parent.mkdir(exist_ok=True)
    joblib.dump(modelo, conj.artefato, compress=3)
    print(f"modelo final de '{conj.nome}' ({len(tv)} fotos de treino + validacao): {conj.artefato}")
    return modelo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("etapa", choices=["mascaras", "extrair", "cv", "final"])
    adicionar_opcao_dataset(ap)
    args = ap.parse_args(argv)
    conj = carregar_conjunto(args.dataset)
    conj.pasta.mkdir(parents=True, exist_ok=True)
    if args.etapa == "mascaras":
        gerar_mascaras_do_conjunto(conj)
    elif args.etapa == "extrair":
        print(f"{conj.nome}: extraindo circulos e atributos de {len(conj.meta)} fotos...", flush=True)
        dados = atributos_do_conjunto(conj)
        print(f"cache: {conj.cache(rh.PARAM1, rh.PARAM2)}")
        print("\ncirculos por classe (so o TREINO):")
        print(resumo_propostas(conj.meta, dados).round(3).to_string())
        figura_propostas(conj, conj.pasta / "propostas_treino.jpg")
        print(f"figura: {conj.pasta / 'propostas_treino.jpg'}")
    elif args.etapa == "cv":
        rodar_cv(conj)
    elif args.etapa == "final":
        ajustar_final(conj)


if __name__ == "__main__":
    main()
