"""Modelo de ferrugem do RoCoLe sobre a transformada de Hough: extrai os atributos, valida e treina.

Uso (na raiz do repositorio, com PYTHONPATH=src):
  python -P src/hough_treinar.py mascaras   # SAM nas fotos de baixa confianca (so as que folha_sam.py recusou)
  python -P src/hough_treinar.py extrair    # circulos do Hough e atributos de todas as fotos (CPU, em paralelo)
  python -P src/hough_treinar.py cv         # validacao cruzada (3 dobras por planta, treino + validacao)
  python -P src/hough_treinar.py final      # ajusta em treino + validacao e grava models/rocole_hough.joblib
  python -P src/hough_treinar.py fusao      # fusao tardia com a ResNet-50, escolhida fora da dobra (GPU)
  python -P src/hough_treinar.py cv-repetida  # auditoria: repete a CV com outras dobras (CPU, ~25 min)

Todas as etapas aceitam --dataset rocole|bracol (ou --rocole, --bracol); sem a opcao, e o RoCoLe. Cada conjunto tem o
seu modelo, treinado e testado nele mesmo:
  python -P src/hough_treinar.py mascaras --bracol   # SAM em todas as folhas do BRACOL (GPU)
  python -P src/hough_treinar.py extrair --bracol
  python -P src/hough_treinar.py cv --bracol         # resultados em outputs/hough_bracol/
  python -P src/hough_treinar.py final --bracol      # grava models/bracol_hough.joblib

Pergunta: ha ferrugem na foto? Positivo = rust; negativo = healthy e red_spider_mite. Tudo e escolhido em treino +
validacao; o teste so e lido por hough_avaliar.py.
"""
import argparse
import io
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

import hough as rh
from comum import (CLASSES_ROCOLE, MODELOS_ROCOLE, OUT_ROCOLE, ROCOLE_CACHE_HOUGH, ROCOLE_MASCARAS, ROCOLE_ZIP,
                          SEED)
from hough_dados import MASCARAS_BAIXA, adicionar_opcao_dataset, carregar_conjunto

PASTA = OUT_ROCOLE / "hough"  # resultados do RoCoLe (os outros conjuntos tem a sua pasta: hough_dados.py)
ARTEFATO = MODELOS_ROCOLE / "rocole_hough.joblib"


# ---------------------------------------------------------------- extracao em lote
def pastas_mascaras():
    return [ROCOLE_MASCARAS, MASCARAS_BAIXA]


def carregar_mascara(id_, forma, pastas=None):
    """Mascara da folha em cache (<pasta>/<id>.png, 960x540), ampliada para forma (A, L). None se nao ha."""
    for pasta in pastas_mascaras() if pastas is None else pastas:
        arq = Path(pasta) / f"{id_}.png"
        if arq.exists():
            m = cv2.imread(str(arq), cv2.IMREAD_GRAYSCALE)
            return cv2.resize(m, (forma[1], forma[0]), interpolation=cv2.INTER_NEAREST) > 0
    return None


def confiancas_sam(arq=None):
    """id -> confianca da mascara do SAM (outputs/folhas_sam.csv); fotos sem mascara valida ficam de fora."""
    arq = arq or OUT_ROCOLE / "folhas_sam.csv"
    if not Path(arq).exists():
        return {}
    t = pd.read_csv(arq).dropna(subset=["confianca"])
    return dict(zip(t["id"].astype(str), t["confianca"].astype(float)))


_ZIP = None


def _abrir_zip(caminho):
    global _ZIP
    cv2.setNumThreads(1)  # um processo por nucleo
    _ZIP = zipfile.ZipFile(caminho)


def _sem_zip():
    cv2.setNumThreads(1)


def _ler_foto(arquivo, z=None):
    """Foto BGR: de dentro do zip z (RoCoLe) ou, sem zip, do arquivo solto (BRACOL)."""
    if z is not None:
        return cv2.imdecode(np.frombuffer(z.read(arquivo), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.imread(str(arquivo), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"nao consegui ler a foto: {arquivo}")
    return img


def _extrair_uma(tarefa):
    id_, arquivo, pastas, confianca, param1, param2 = tarefa
    img = _ler_foto(arquivo, _ZIP)
    m = carregar_mascara(id_, img.shape[:2], pastas)
    r = rh.extrair(img, m, confianca if m is not None else 0.0, param1, param2)
    return id_, {"circulos": r["circulos"], "at_circulos": r["at_circulos"], "at_folha": r["at_folha"]}


def extrair_tudo(meta, caminho_zip, param1=rh.PARAM1, param2=rh.PARAM2, pastas=None, confiancas=None, processos=None,
                 cache=None):
    """Circulos e atributos de todas as fotos de meta (colunas id, arquivo): id -> {"circulos", "at_circulos",
    "at_folha"}. Os atributos nao usam o rotulo. Com `cache` (arquivo .joblib), grava e reaproveita.
    caminho_zip None: 'arquivo' e o caminho de cada foto no disco (BRACOL)."""
    if cache is not None and Path(cache).exists():
        return joblib.load(cache)
    confiancas = confiancas_sam() if confiancas is None else confiancas
    pastas = [str(p) for p in (pastas_mascaras() if pastas is None else pastas)]
    tarefas = [(str(r.id), r.arquivo, pastas, float(confiancas.get(str(r.id), 0.0)), param1, param2)
               for r in meta.itertuples()]
    processos = processos or max(1, (os.cpu_count() or 2) - 1)
    if processos == 1:
        global _ZIP
        if caminho_zip is not None:
            _abrir_zip(str(caminho_zip))
        try:
            pares = [_extrair_uma(t) for t in tarefas]
        finally:  # no Windows, o zip aberto nao pode ser apagado nem substituido
            if _ZIP is not None:
                _ZIP.close()
                _ZIP = None
    else:
        iniciar, args = (_abrir_zip, (str(caminho_zip),)) if caminho_zip is not None else (_sem_zip, ())
        with Pool(processos, initializer=iniciar, initargs=args) as pool:
            pares = pool.map(_extrair_uma, tarefas, chunksize=8)
    dados = dict(pares)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dados, cache, compress=3)
    return dados


def arquivo_cache(param1=rh.PARAM1, param2=rh.PARAM2):
    return ROCOLE_CACHE_HOUGH / f"atributos_p{param1}_{param2}.joblib"


def atributos_do_conjunto(conj, param1=rh.PARAM1, param2=rh.PARAM2):
    """Circulos e atributos de todas as fotos de um conjunto (hough_dados.Conjunto), do cache dele se ja
    foram extraidos."""
    return extrair_tudo(conj.meta, conj.zip, param1, param2, pastas=conj.pastas_mascaras,
                        confiancas=confiancas_sam(conj.arquivo_confiancas), cache=conj.cache(param1, param2))


def gerar_mascaras_do_conjunto(conj):
    """SAM (ponto no centro) em todas as fotos de um conjunto que nao tem mascaras prontas (BRACOL): grava a mascara
    de cada foto (metade da resolucao) e a confianca em conj.arquivo_confiancas. Retoma de onde parou."""
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
        img = _ler_foto(r.arquivo, None)
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


def gerar_mascaras_baixa_confianca(meta, caminho_zip):
    """SAM (ponto no centro) nas fotos que folha_sam.py recusou: grava em MASCARAS_BAIXA as que tem area valida,
    com qualquer confianca. As aceitas continuam em ROCOLE_MASCARAS."""
    import torch
    from PIL import Image

    from folha_sam import SegmentadorSAM
    MASCARAS_BAIXA.mkdir(parents=True, exist_ok=True)
    faltam = [r for r in meta.itertuples() if not (ROCOLE_MASCARAS / f"{r.id}.png").exists()]
    print(f"{len(faltam)} fotos sem mascara aceita; rodando o SAM...", flush=True)
    sam = SegmentadorSAM(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    z = zipfile.ZipFile(caminho_zip)
    feitas = 0
    for r in faltam:
        img = np.asarray(Image.open(io.BytesIO(z.read(r.arquivo))).convert("RGB"))
        m, _ = rh.escolher_mascara_hough(*sam(img))
        if m is not None:
            h, w = m.shape
            Image.fromarray(m.astype(np.uint8) * 255).resize((w // 2, h // 2), Image.NEAREST).save(
                MASCARAS_BAIXA / f"{r.id}.png")
            feitas += 1
    print(f"mascaras de baixa confianca gravadas: {feitas} de {len(faltam)} (as outras usam o contorno pela cor)")


def figura_propostas(conj, caminho, n=12, param1=rh.PARAM1, param2=rh.PARAM2):
    """Fotos de TREINO sorteadas, com o contorno da folha e os circulos do Hough (vermelho = os 15 de mais votos),
    ao lado do mapa de lesao. So para conferir a olho."""
    z = zipfile.ZipFile(conj.zip) if conj.zip is not None else None
    conf = confiancas_sam(conj.arquivo_confiancas)
    meta = conj.meta
    amostra = meta[meta["split"] == "train"].groupby("classe", group_keys=False).sample(n // 3, random_state=SEED)
    tiles = []
    for r in amostra.itertuples():
        img = _ler_foto(r.arquivo, z)
        m = carregar_mascara(r.id, img.shape[:2], conj.pastas_mascaras)
        e = rh.extrair(img, m, conf.get(str(r.id), 0.0) if m is not None else 0.0, param1, param2)
        vis = e["folha"].img.copy()
        cont, _ = cv2.findContours(e["folha"].mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cont, -1, (255, 0, 255), 2)
        for i, (x, y, raio) in enumerate(e["circulos"]):
            cv2.circle(vis, (int(x), int(y)), int(raio), (0, 0, 255) if i < 15 else (0, 255, 255), 2)
        par = np.hstack([vis, cv2.cvtColor(e["mapa"], cv2.COLOR_GRAY2BGR)])
        par = cv2.resize(par, (900, max(1, int(par.shape[0] * 900 / par.shape[1]))))
        par = cv2.copyMakeBorder(par, 0, max(0, 400 - par.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:400]
        cv2.putText(par, f"{r.id} {r.classe} {r.lado} n={len(e['circulos'])}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)
        tiles.append(par)
    linhas = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
    cv2.imwrite(str(caminho), np.vstack(linhas), [cv2.IMWRITE_JPEG_QUALITY, 85])


def resumo_propostas(meta, dados):
    """Circulos por classe e lado no TREINO: media, mediana e fracao das fotos com pelo menos 1 circulo."""
    tr = meta[meta["split"] == "train"]
    n = tr["id"].map(lambda i: len(dados[i]["circulos"]))
    return tr.assign(n=n).groupby(["classe", "lado"])["n"].agg(media="mean", mediana="median",
                                                                 com_circulo=lambda s: (s > 0).mean())


# ---------------------------------------------------------------- modelo de dois niveis
CLASSES = CLASSES_ROCOLE  # healthy, red_spider_mite, rust
RUST = CLASSES.index("rust")
N_DOBRAS, N_INTERNAS = 3, 3
VARIANTES = ("completo", "so_folha", "so_circulos", "rico")
GRADE_LIMIAR = np.round(np.arange(0.20, 0.801, 0.02), 2)
_CURTO = {"healthy": "healthy", "red_spider_mite": "mite", "rust": "rust"}
NOMES_SACO = (["n_circulos"] + [f"{e}_{_CURTO[c]}" for c in CLASSES for e in ("max", "top3", "top5", "media", "soma")]
              + ["n_rust_05", "n_rust_07", "area_rust"])


def estatisticas_saco(p, raios, area_folha=rh.LADO_FOLHA ** 2):
    """Resume as pontuacoes dos circulos de uma foto (p: (N, 3), nivel 1) num vetor fixo, na ordem de NOMES_SACO:
    quantos circulos; por classe, maximo, media dos 3 e dos 5 maiores, media e soma; quantos parecem ferrugem
    (> 0,5 e > 0,7); e a area dos circulos pesada pela pontuacao de ferrugem, em fracao da folha. Sem circulos: zeros."""
    p = np.asarray(p, np.float64).reshape(-1, len(CLASSES))
    if not len(p):
        return np.zeros(len(NOMES_SACO), np.float32)
    saida = [len(p)]
    for c in range(len(CLASSES)):
        ordenado = np.sort(p[:, c])[::-1]
        saida += [ordenado[0], ordenado[:3].mean(), ordenado[:5].mean(), ordenado.mean(), ordenado.sum()]
    r = np.asarray(raios, np.float64)
    saida += [(p[:, RUST] > 0.5).sum(), (p[:, RUST] > 0.7).sum(), (p[:, RUST] * np.pi * r ** 2).sum() / area_folha]
    return np.asarray(saida, np.float32)


ATRIBUTOS_RICOS = ["raio", "apoio", "croma", "mapa_media", "delta_L", "delta_a", "delta_b", "centro_L", "variancia",
                   "blob_area", "blob_circularidade", "blob_alongamento"]
_IDX_RICOS = [rh.NOMES_CIRCULO.index(n) for n in ATRIBUTOS_RICOS]
NOMES_RICOS = ([f"top5_{n}" for n in ATRIBUTOS_RICOS] + [f"todos_{n}" for n in ATRIBUTOS_RICOS]
               + [f"hist_rust_{i}" for i in range(5)])


def estatisticas_ricas(p, at_circulos):
    """Alem das pontuacoes, COMO sao os circulos da foto (ordem de NOMES_RICOS): media de ATRIBUTOS_RICOS nos 5
    circulos mais parecidos com ferrugem e em todos, e o histograma de P(ferrugem) dos circulos em 5 faixas."""
    p = np.asarray(p, np.float64).reshape(-1, len(CLASSES))
    if not len(p):
        return np.zeros(len(NOMES_RICOS), np.float32)
    at = np.asarray(at_circulos, np.float64)[:, _IDX_RICOS]
    top = np.argsort(-p[:, RUST], kind="stable")[:5]
    hist = np.bincount(np.minimum((p[:, RUST] * 5).astype(int), 4), minlength=5) / len(p)
    return np.concatenate([at[top].mean(0), at.mean(0), hist]).astype(np.float32)


def montar_x(ids, dados, p_circ, variante):
    """Matriz do nivel 2: estatisticas do saco de circulos e/ou atributos da folha, conforme a variante.
    p_circ: id -> (N, 3) com a pontuacao dos circulos (nao usado em 'so_folha')."""
    partes = []
    if variante in ("completo", "so_circulos", "rico"):
        partes.append(np.stack([estatisticas_saco(p_circ[i], dados[i]["circulos"][:, 2]) for i in ids]))
    if variante == "rico":
        partes.append(np.stack([estatisticas_ricas(p_circ[i], dados[i]["at_circulos"]) for i in ids]))
    if variante in ("completo", "so_folha", "rico"):
        partes.append(np.stack([dados[i]["at_folha"] for i in ids]))
    if not partes:
        raise ValueError(f"variante desconhecida: {variante}")
    return np.hstack(partes).astype(np.float32)


def _nivel1():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)


MediaDeModelos = rh.MediaDeModelos  # a classe mora em hough.py: este arquivo roda como script, e uma classe
# definida num script e gravada no modelo como __main__.MediaDeModelos, que nenhum outro programa consegue carregar


def _nivel2(tipo="hgb"):
    """hgb: gradient boosting; logistica: regressao logistica padronizada; floresta: arvores extremamente
    aleatorias; media: media das probabilidades de hgb e logistica."""
    if tipo == "hgb":
        return _nivel1()
    if tipo == "floresta":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=SEED,
                                    n_jobs=-1)
    if tipo == "media":
        return MediaDeModelos([_nivel2("hgb"), _nivel2("logistica")])
    if tipo != "logistica":
        raise ValueError(f"nivel 2 desconhecido: {tipo}")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def _circulos_de(ids, y, dados):
    """Empilha os circulos das fotos: atributos e o rotulo da FOTO de cada um (rotulo fraco)."""
    x = np.vstack([dados[i]["at_circulos"] for i in ids])
    yc = np.concatenate([np.full(len(dados[i]["at_circulos"]), c) for i, c in zip(ids, y)])
    return x, yc


def _pontuar(modelo, ids, dados):
    """id -> (N, 3): probabilidade de cada circulo ser de foto saudavel, de acaro ou de ferrugem."""
    saida = {}
    for i in ids:
        at = dados[i]["at_circulos"]
        p = np.zeros((len(at), len(CLASSES)))
        if len(at):
            p[:, modelo.classes_] = modelo.predict_proba(at)
        saida[i] = p
    return saida


def pontuar_fora_da_dobra(ids, y, grupos, dados, fabrica=_nivel1, n_dobras=N_INTERNAS):
    """Pontua os circulos de cada foto com um nivel 1 que nao viu nenhuma foto do mesmo grupo (planta)."""
    from sklearn.model_selection import StratifiedGroupKFold
    ids, y, grupos = np.asarray(ids), np.asarray(y), np.asarray(grupos)
    saida = {}
    sgkf = StratifiedGroupKFold(n_splits=n_dobras, shuffle=True, random_state=SEED)
    for tr, va in sgkf.split(ids, y, grupos):
        x, yc = _circulos_de(ids[tr], y[tr], dados)
        saida.update(_pontuar(fabrica().fit(x, yc), ids[va], dados))
    return saida


def _indices(classes, nomes=None):
    """Indice de cada classe em `nomes` (os nomes das classes do conjunto; padrao: os do RoCoLe). Inteiros passam
    direto. Nome desconhecido: ValueError."""
    nomes = list(CLASSES if nomes is None else nomes)
    return np.array([nomes.index(c) if isinstance(c, str) else int(c) for c in classes])


def ajustar_nivel1(ids, classes, grupos, dados, nomes=None):
    """Nivel 1 ajustado em todas as fotos de ids, e a pontuacao FORA DA DOBRA (por planta) dos circulos dessas
    mesmas fotos, que e o que o nivel 2 pode ver no treino."""
    ids, y = list(ids), _indices(classes, nomes)
    return _nivel1().fit(*_circulos_de(ids, y, dados)), pontuar_fora_da_dobra(ids, y, grupos, dados)


def ajustar(ids, classes, grupos, dados, variante="completo", tipo_nivel2="hgb", nivel1=None, p_circ=None, nomes=None):
    """Treina os dois niveis. O nivel 2 so ve pontuacoes de circulos feitas fora da dobra, por planta.
    classes: nomes (os de `nomes`; padrao CLASSES, do RoCoLe) ou indices. nivel1 e p_circ: resultado de
    ajustar_nivel1, para nao refazer. Devolve {"nivel1", "nivel2", "variante"}."""
    ids, y = list(ids), _indices(classes, nomes)
    if variante == "so_folha":
        nivel1, p_circ = None, None
    elif nivel1 is None:
        nivel1, p_circ = ajustar_nivel1(ids, y, grupos, dados)
    nivel2 = _nivel2(tipo_nivel2).fit(montar_x(ids, dados, p_circ, variante), y)
    return {"nivel1": nivel1, "nivel2": nivel2, "variante": variante}


def prever(modelo, ids, dados):
    """Probabilidades (n, 3), na ordem de CLASSES."""
    ids = list(ids)
    p_circ = _pontuar(modelo["nivel1"], ids, dados) if modelo["nivel1"] is not None else None
    n2 = modelo["nivel2"]
    p = np.zeros((len(ids), len(CLASSES)))
    p[:, n2.classes_] = n2.predict_proba(montar_x(ids, dados, p_circ, modelo["variante"]))
    return p


def dobras(meta, n_dobras=N_DOBRAS, seed=SEED):
    """Dobra (0..n-1) de cada foto de treino + validacao, indexada pelo id: as mesmas plantas de fora de
    cnn_cv.dobras_cv (copiado aqui para nao importar o torch)."""
    import warnings

    from sklearn.model_selection import StratifiedGroupKFold
    base = meta[meta["split"].isin(["train", "val"])].reset_index(drop=True)
    d = pd.Series(-1, index=base["id"].to_numpy(), name="dobra")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sgkf = StratifiedGroupKFold(n_splits=n_dobras, shuffle=True, random_state=seed)
        for k, (_, fora) in enumerate(sgkf.split(base, base["classe"], base["grupo"])):
            d.iloc[fora] = k
    return d


def cv_externa(meta, dados, variante="completo", tipo_nivel2="hgb", niveis1=None, semente=SEED, nomes=None):
    """Predicoes fora da dobra de todas as fotos de treino + validacao: id, dobra, classe, grupo, p_<classe>.
    niveis1: dicionario reaproveitado entre chamadas (dobra -> nivel 1 e pontuacoes), para comparar variantes do
    nivel 2 sem refazer o nivel 1, que e a parte demorada. semente: a das dobras (SEED = as de cnn_cv.py)."""
    nomes = list(CLASSES if nomes is None else nomes)
    d = dobras(meta, seed=semente)
    base = meta[meta["split"].isin(["train", "val"])].set_index("id").loc[d.index]
    niveis1 = {} if niveis1 is None else niveis1
    partes = []
    for k in range(N_DOBRAS):
        tr, fora = base[(d != k).to_numpy()], base[(d == k).to_numpy()]
        if variante != "so_folha" and k not in niveis1:
            niveis1[k] = ajustar_nivel1(tr.index, tr["classe"], tr["grupo"], dados, nomes)
        n1, p_circ = niveis1.get(k, (None, None))
        modelo = ajustar(tr.index, tr["classe"], tr["grupo"], dados, variante, tipo_nivel2, n1, p_circ, nomes)
        p = prever(modelo, fora.index, dados)
        partes.append(pd.DataFrame({"id": fora.index, "dobra": k, "classe": fora["classe"].to_numpy(),
                                    "grupo": fora["grupo"].to_numpy(),
                                    **{f"p_{c}": p[:, j] for j, c in enumerate(nomes)}}))
    return pd.concat(partes, ignore_index=True)


def _f1(y, pred):
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def escolher_limiar(y_rust, p_rust, grade=GRADE_LIMIAR):
    """Limiar de P(ferrugem) com o maior F1; no empate, o mais perto de 0,5. Sem nenhum acerto possivel: 0,5."""
    y, p = np.asarray(y_rust, bool), np.asarray(p_rust, float)
    f1 = np.array([_f1(y, p >= t) for t in grade])
    if f1.max() <= 0:
        return 0.5
    melhores = grade[f1 >= f1.max() - 1e-12]
    return float(melhores[np.abs(melhores - 0.5).argmin()])


def f1_ferrugem_por_dobra(oof, limiar=None):
    """F1 de ferrugem em cada dobra. limiar None: cada dobra e pontuada com o limiar escolhido nas OUTRAS dobras
    (o numero honesto); com um valor, usa esse limiar em todas."""
    y = (oof["classe"] == "rust").to_numpy()
    p, d = oof["p_rust"].to_numpy(), oof["dobra"].to_numpy()
    saida = {}
    for k in sorted(np.unique(d)):
        t = escolher_limiar(y[d != k], p[d != k]) if limiar is None else limiar
        saida[int(k)] = _f1(y[d == k], p[d == k] >= t)
    return pd.Series(saida, name="f1_ferrugem")


def decidir(tab_f1, base, candidata):
    """Regra fixada antes de rodar: a candidata substitui a base so se a media do F1 for maior E se ela vencer em
    pelo menos 2 das 3 dobras. tab_f1: linhas = dobras, colunas = variantes."""
    return bool(tab_f1[candidata].mean() > tab_f1[base].mean() and (tab_f1[candidata] > tab_f1[base]).sum() >= 2)


def resumo_oof(oof):
    """Metricas fora da dobra de uma variante: F1 de ferrugem por dobra (limiar das outras dobras) e media, e, com
    o limiar unico escolhido em todas as dobras, sensibilidade, especificidade e F1; mais o F1 macro de 3 classes."""
    from sklearn.metrics import f1_score
    f1 = f1_ferrugem_por_dobra(oof)
    y = (oof["classe"] == "rust").to_numpy()
    t = escolher_limiar(y, oof["p_rust"].to_numpy())
    pred = oof["p_rust"].to_numpy() >= t
    nomes = [c[2:] for c in oof.columns if c.startswith("p_")]  # as classes do conjunto, na ordem dos indices
    pred3 = np.array(nomes)[oof[[f"p_{c}" for c in nomes]].to_numpy().argmax(1)]
    return {**{f"f1_d{k}": v for k, v in f1.items()}, "f1_media": float(f1.mean()), "limiar": t,
            "f1_limiar_unico": _f1(y, pred), "sensibilidade": float(pred[y].mean()),
            "especificidade": float((~pred[~y]).mean()),
            "f1_macro_3_classes": float(f1_score(oof["classe"], pred3, average="macro"))}


def _linha_resnet(meta):
    """ResNet-50 (foto inteira) fora da dobra, das predicoes ja gravadas por cnn_cv.py: mesma tabela."""
    from sklearn.metrics import f1_score
    arq = OUT_ROCOLE / "cv_entradas" / "predicoes_fora_da_dobra.csv"
    if not arq.exists():
        return None
    r = pd.read_csv(arq)
    r = r[r["entrada"] == "foto"]
    y, pred = (r["classe"] == "rust").to_numpy(), (r["classe_predita"] == "rust").to_numpy()
    f1 = {int(k): _f1(y[(r["dobra"] == k).to_numpy()], pred[(r["dobra"] == k).to_numpy()]) for k in sorted(r["dobra"].unique())}
    return {"variante": "resnet50_foto (referencia, sem limiar)", **{f"f1_d{k}": v for k, v in f1.items()},
            "f1_media": float(np.mean(list(f1.values()))), "limiar": float("nan"), "f1_limiar_unico": _f1(y, pred),
            "sensibilidade": float(pred[y].mean()), "especificidade": float((~pred[~y]).mean()),
            "f1_macro_3_classes": float(f1_score(r["classe"], r["classe_predita"], average="macro"))}


CONFIGS = [  # (nome, param1, param2, variante, nivel 2); a primeira e a base
    ("completo", rh.PARAM1, rh.PARAM2, "completo", "hgb"),
    ("so_folha (sem Hough)", rh.PARAM1, rh.PARAM2, "so_folha", "hgb"),
    ("so_circulos", rh.PARAM1, rh.PARAM2, "so_circulos", "hgb"),
    ("completo_logistica", rh.PARAM1, rh.PARAM2, "completo", "logistica"),
    ("completo_media", rh.PARAM1, rh.PARAM2, "completo", "media"),
    ("completo_floresta", rh.PARAM1, rh.PARAM2, "completo", "floresta"),
    ("rico", rh.PARAM1, rh.PARAM2, "rico", "hgb"),
    ("rico_media", rh.PARAM1, rh.PARAM2, "rico", "media"),
    ("rico_floresta", rh.PARAM1, rh.PARAM2, "rico", "floresta"),
    ("completo_hough_100_20", 100, 20, "completo", "hgb"),
]


def rodar_cv(conj):
    """Validacao cruzada das variantes em CONFIGS no conjunto dado; grava cv_oof.csv, cv_resumo.csv e decisao.json
    na pasta do conjunto."""
    import json
    meta, PASTA = conj.meta, conj.pasta
    PASTA.mkdir(parents=True, exist_ok=True)
    linhas, oofs, niveis1, atributos = [], [], {}, {}
    for nome, p1, p2, variante, tipo in CONFIGS:
        if (p1, p2) != (rh.PARAM1, rh.PARAM2) and not conj.cache(p1, p2).exists():
            print(f"pulando '{nome}': sem atributos com param1 {p1} e param2 {p2} para '{conj.nome}' (rode: extrair "
                  f"--dataset {conj.nome} --param1 {p1} --param2 {p2})", flush=True)
            continue
        if (p1, p2) not in atributos:
            atributos[(p1, p2)] = atributos_do_conjunto(conj, p1, p2)
        dados = atributos[(p1, p2)]
        oof = cv_externa(meta, dados, variante, tipo, niveis1.setdefault((p1, p2), {}), nomes=conj.classes)
        oofs.append(oof.assign(variante=nome))
        linhas.append({"variante": nome, **resumo_oof(oof)})
        print(f"{nome}: F1 de ferrugem por dobra {[round(linhas[-1][f'f1_d{k}'], 3) for k in range(N_DOBRAS)]} | media "
              f"{linhas[-1]['f1_media']:.4f}", flush=True)
    tab = pd.DataFrame(linhas)
    f1 = tab.set_index("variante")[[f"f1_d{k}" for k in range(N_DOBRAS)]].T.reset_index(drop=True)
    escolhida = CONFIGS[0][0]
    aprovadas = {c: float(f1[c].mean()) for c in f1.columns[1:]
                 if not c.startswith("so_") and decidir(f1, CONFIGS[0][0], c)}
    if aprovadas:
        escolhida = max(aprovadas, key=aprovadas.get)
    ref = _linha_resnet(meta) if conj.nome == "rocole" else None
    if ref:
        tab = pd.concat([tab, pd.DataFrame([ref])], ignore_index=True)
    pd.concat(oofs, ignore_index=True).to_csv(PASTA / "cv_oof.csv", index=False)
    tab.to_csv(PASTA / "cv_resumo.csv", index=False)
    cfg = next(c for c in CONFIGS if c[0] == escolhida)
    linha = tab[tab["variante"] == escolhida].iloc[0]
    decisao = {"conjunto": conj.nome, "escolhida": escolhida, "param1": cfg[1], "param2": cfg[2], "variante": cfg[3],
               "nivel2": cfg[4],
               "limiar": float(linha["limiar"]), "f1_media": float(linha["f1_media"]),
               "aprovadas_pela_regra": sorted(aprovadas),
               "regra": "substitui a base 'completo' so com media maior E vitoria em >= 2 das 3 dobras; as variantes "
                        "so_folha e so_circulos sao ablacoes e nao concorrem"}
    with open(PASTA / "decisao.json", "w", encoding="utf-8") as f:
        json.dump(decisao, f, indent=2)
    print(f"\n--- {conj.nome}: fora da dobra (3 dobras por grupo, treino + validacao; F1 de ferrugem, foto a foto) ---")
    print(tab.round(4).to_string(index=False))
    print(f"\nescolhida: {escolhida} | limiar {decisao['limiar']:.2f} | gravado: {PASTA / 'decisao.json'}")
    return decisao


def ajustar_final(conj):
    """Ajusta a configuracao escolhida em treino + validacao do conjunto e grava o modelo dele (conj.artefato), com o
    limiar da validacao cruzada."""
    import json
    meta, ARTEFATO = conj.meta, conj.artefato
    arq = conj.pasta / "decisao.json"
    if not arq.exists():
        raise SystemExit(f"{arq} nao existe: rode hough_treinar.py cv --dataset {conj.nome}")
    with open(arq, encoding="utf-8") as f:
        dec = json.load(f)
    dados = atributos_do_conjunto(conj, dec["param1"], dec["param2"])
    tv = meta[meta["split"].isin(["train", "val"])]
    modelo = ajustar(tv["id"], tv["classe"], tv["grupo"], dados, dec["variante"], dec["nivel2"], nomes=conj.classes)
    modelo.update({"param1": dec["param1"], "param2": dec["param2"], "limiar": dec["limiar"],
                   "classes": list(conj.classes), "conjunto": conj.nome,
                   "nomes_circulo": rh.NOMES_CIRCULO, "nomes_folha": rh.NOMES_FOLHA, "nomes_saco": NOMES_SACO,
                   "fusao": _fusao_do_artefato(dec.get("fusao")),
                   "recusa": _recusa_do_artefato(dec.get("recusa")),
                   "metadados_sha256": conj.sha256()})
    ARTEFATO.parent.mkdir(exist_ok=True)
    joblib.dump(modelo, ARTEFATO, compress=3)
    print(f"modelo final de '{conj.nome}' ({dec['escolhida']}, {len(tv)} fotos de treino + validacao, limiar "
          f"{dec['limiar']:.2f}): {ARTEFATO}")
    return modelo


# ---------------------------------------------------------------- validacao cruzada repetida (auditoria)
def resumo_repetida(tab, base):
    """tab: variante, repeticao, dobra, f1. Para cada variante: media das repeticoes, desvio-padrao entre
    repeticoes, menor e maior repeticao, diferenca media para a base (pareada por repeticao), em quantas
    repeticoes a media vence a da base e em que fracao das dobras (pareadas) ela vence."""
    por_rep = tab.pivot_table(index="repeticao", columns="variante", values="f1", aggfunc="mean")
    por_dobra = tab.pivot_table(index=["repeticao", "dobra"], columns="variante", values="f1")
    linhas = []
    for v in por_rep.columns:
        dif = por_rep[v] - por_rep[base]
        linhas.append({"variante": v, "media": float(por_rep[v].mean()),
                       "dp_repeticoes": float(por_rep[v].std(ddof=1)) if len(por_rep) > 1 else float("nan"),
                       "min_rep": float(por_rep[v].min()), "max_rep": float(por_rep[v].max()),
                       "dif_base": float(dif.mean()), "repeticoes_ganhas": int((dif > 1e-12).sum()),
                       "repeticoes": int(len(por_rep)),
                       "dobras_ganhas": float((por_dobra[v] > por_dobra[base] + 1e-12).mean())})
    return pd.DataFrame(linhas).sort_values("media", ascending=False).reset_index(drop=True)


def rodar_cv_repetida(conj, repeticoes=5):
    """Repete a validacao cruzada por planta com outras sementes de dobras (a repeticao 0 e a de cnn_cv.py),
    para ver se a ordem das variantes do modelo de Hough se mantem. E uma AUDITORIA: nao troca o modelo escolhido
    nem toca no teste. Grava cv_repetida.csv a cada repeticao (retoma de onde parou) e cv_repetida_resumo.csv."""
    meta, PASTA = conj.meta, conj.pasta
    arq = PASTA / "cv_repetida.csv"
    tab = pd.read_csv(arq) if arq.exists() else pd.DataFrame(columns=["variante", "repeticao", "semente", "dobra", "f1"])
    configs = [c for c in CONFIGS if (c[1], c[2]) == (rh.PARAM1, rh.PARAM2)]
    dados = atributos_do_conjunto(conj)
    for r in range(repeticoes):
        if (tab["repeticao"] == r).any():
            print(f"repeticao {r}: ja feita", flush=True)
            continue
        niveis1, linhas = {}, []
        for nome, _, _, variante, tipo in configs:
            f1 = f1_ferrugem_por_dobra(cv_externa(meta, dados, variante, tipo, niveis1, semente=SEED + r,
                                                  nomes=conj.classes))
            linhas += [{"variante": nome, "repeticao": r, "semente": SEED + r, "dobra": k, "f1": v} for k, v in f1.items()]
        tab = pd.concat([tab, pd.DataFrame(linhas)], ignore_index=True)
        tab.to_csv(arq, index=False)
        medias = pd.DataFrame(linhas).groupby("variante")["f1"].mean()
        print(f"repeticao {r} (semente {SEED + r}): " + " | ".join(f"{n} {v:.3f}" for n, v in medias.items()), flush=True)
    resumo = resumo_repetida(tab.astype({"repeticao": int, "dobra": int, "f1": float}), base=configs[0][0])
    resumo.to_csv(PASTA / "cv_repetida_resumo.csv", index=False)
    print()
    print(f"--- Validacao cruzada repetida ({resumo['repeticoes'].iloc[0]} repeticoes x {N_DOBRAS} dobras; F1 de ferrugem, "
          f"limiar das outras dobras; base = '{configs[0][0]}') ---")
    print(resumo.round(4).to_string(index=False))
    return resumo


# ---------------------------------------------------------------- fusao tardia com a ResNet-50
PESOS_FUSAO = (0.25, 0.5, 0.75)  # peso do modelo de Hough na media das probabilidades
ARQ_RESNET_OOF = PASTA / "resnet50_oof.csv"


def fundir(p_b, p_cnn, w):
    """Fusao tardia: w * P(Hough) + (1 - w) * P(ResNet-50). Nenhum modelo e ajustado sobre as saidas da ResNet, e
    por isso as predicoes fora da dobra continuam limpas."""
    return w * np.asarray(p_b, float) + (1.0 - w) * np.asarray(p_cnn, float)


def _prever_resnet_dobra(k, ids, meta):
    """Probabilidades da ResNet-50 da dobra k (models/cv/resnet50_foto_d<k>.pt, treinada por cnn_cv.py sem as
    plantas da dobra k) para as fotos ids: id -> (3,)."""
    import torch
    from torch.utils.data import DataLoader

    from cnn_avaliar import prever as prever_cnn
    from cnn_modelo import FolhasRocole, construir_modelo
    pesos = MODELOS_ROCOLE / "cv" / f"resnet50_foto_d{k}.pt"
    if not pesos.exists():
        raise SystemExit(f"{pesos} nao existe: rode cnn_cv.py (modelos das dobras)")
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = construir_modelo("resnet50", len(CLASSES), pretreinado=False)
    m.load_state_dict(torch.load(pesos, map_location=disp, weights_only=True))
    sub = meta[meta["id"].isin(set(ids))].assign(split="fora")
    probs, _, ids_p = prever_cnn(m.to(disp).eval(), DataLoader(FolhasRocole("fora", meta=sub), batch_size=32), disp)
    return dict(zip(ids_p, probs))


def probs_resnet_oof(meta, prever_dobra=None):
    """Probabilidades da ResNet-50 fora da dobra para treino + validacao: cada foto e prevista so pelo modelo da
    sua dobra, que nao viu a planta dela. Colunas: id, dobra, p_<classe>."""
    d = dobras(meta)
    prever_dobra = prever_dobra or (lambda k, ids: _prever_resnet_dobra(k, ids, meta))
    linhas = []
    for k in range(N_DOBRAS):
        ids = d.index[d == k].tolist()
        p = prever_dobra(k, ids)
        linhas += [{"id": i, "dobra": k, **{f"p_{c}": float(p[i][j]) for j, c in enumerate(CLASSES)}} for i in ids]
    return pd.DataFrame(linhas)


def tabela_fusao(oof_b, cnn, pesos=PESOS_FUSAO):
    """Compara, fora da dobra, o modelo de Hough, a fusao com cada peso e a ResNet-50 sozinha (com limiar, para a
    comparacao ser justa). As fotos sao casadas pelo id. Devolve (tabela como a de resumo_oof, nome -> predicoes)."""
    cols = [f"p_{c}" for c in CLASSES]
    b, c = oof_b[cols].to_numpy(), cnn.set_index("id").loc[oof_b["id"].to_numpy(), cols].to_numpy()
    oofs = {"hough": oof_b}
    for w in pesos:
        oofs[f"fusao_w{w:.2f}"] = oof_b.assign(**dict(zip(cols, fundir(b, c, w).T)))
    oofs["resnet50_com_limiar"] = oof_b.assign(**dict(zip(cols, c.T)))
    return pd.DataFrame([{"variante": n, **resumo_oof(o)} for n, o in oofs.items()]), oofs


def escolher_fusao(tab):
    """A fusao so entra pela mesma regra: media do F1 maior que a do Hough E vitoria em >= 2 das 3 dobras. Entre
    as aprovadas, a de maior media. A ResNet-50 sozinha nao concorre (o modelo final comeca nos circulos)."""
    f1 = tab.set_index("variante")[[f"f1_d{k}" for k in range(N_DOBRAS)]].T.reset_index(drop=True)
    aprovadas = {c: float(f1[c].mean()) for c in f1.columns if c.startswith("fusao_w") and decidir(f1, "hough", c)}
    if not aprovadas:
        return {"adotada": False}
    melhor = max(aprovadas, key=aprovadas.get)
    return {"adotada": True, "variante": melhor, "w": float(melhor.split("w")[-1]),
            "limiar": float(tab.set_index("variante").loc[melhor, "limiar"])}


def _fusao_do_artefato(fusao):
    """O que o modelo gravado guarda da decisao de fusao: peso e limiar, ou None se nao foi adotada."""
    return {"w": fusao["w"], "limiar": fusao["limiar"]} if fusao and fusao.get("adotada") else None


def _recusa_do_artefato(recusa):
    """O que o modelo gravado guarda da opcao de recusa (hough_recusa.py): taxa e margem, ou None."""
    return {"taxa": recusa["taxa"], "margem": recusa["margem"]} if recusa and recusa.get("adotada") else None


def rodar_fusao(meta):
    """Fusao tardia com a ResNet-50, escolhida fora da dobra. Grava resnet50_oof.csv, fusao_resumo.csv e
    fusao_oof.csv, e registra a decisao em decisao.json e no modelo gravado (sem ajusta-lo de novo)."""
    import json
    arq = PASTA / "decisao.json"
    if not arq.exists() or not (PASTA / "cv_oof.csv").exists():
        raise SystemExit(f"{arq} ou cv_oof.csv nao existe: rode hough_treinar.py cv")
    with open(arq, encoding="utf-8") as f:
        dec = json.load(f)
    oof = pd.read_csv(PASTA / "cv_oof.csv")
    oof_b = oof[oof["variante"] == dec["escolhida"]].drop(columns="variante").reset_index(drop=True)
    if ARQ_RESNET_OOF.exists():
        cnn = pd.read_csv(ARQ_RESNET_OOF)
    else:
        print("ResNet-50 das 3 dobras prevendo as fotos de fora...", flush=True)
        cnn = probs_resnet_oof(meta)
        cnn.to_csv(ARQ_RESNET_OOF, index=False)
    ref = OUT_ROCOLE / "cv_entradas" / "predicoes_fora_da_dobra.csv"
    if ref.exists():  # conferencia: as classes devem bater com as gravadas por cnn_cv.py
        r = pd.read_csv(ref)
        r = r[r["entrada"] == "foto"].set_index("id")["classe_predita"]
        agora = pd.Series(np.array(CLASSES)[cnn[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1)], index=cnn["id"])
        print(f"classes da ResNet-50 iguais as de cnn_cv.py: {(agora == r.reindex(agora.index)).mean():.1%}")
    tab, oofs = tabela_fusao(oof_b, cnn)
    esc = escolher_fusao(tab)
    tab.to_csv(PASTA / "fusao_resumo.csv", index=False)
    if esc["adotada"]:
        oofs[esc["variante"]].to_csv(PASTA / "fusao_oof.csv", index=False)
    dec["fusao"] = {**esc, "pesos_testados": list(PESOS_FUSAO),
                    "regra": "a fusao substitui o modelo de Hough so com media maior E vitoria em >= 2 das 3 dobras"}
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(dec, f, indent=2)
    if ARTEFATO.exists():
        modelo = joblib.load(ARTEFATO)
        modelo["fusao"] = _fusao_do_artefato(esc)
        joblib.dump(modelo, ARTEFATO, compress=3)
    print("\n--- Fusao tardia, fora da dobra (F1 de ferrugem; w = peso do modelo de Hough) ---")
    print(tab.round(4).to_string(index=False))
    print(f"\nfusao adotada: {esc['adotada']}" + (f" | w = {esc['w']:.2f} | limiar {esc['limiar']:.2f}" if esc["adotada"] else ""))
    return esc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("etapa", choices=["mascaras", "extrair", "cv", "final", "fusao", "cv-repetida"])
    adicionar_opcao_dataset(ap)
    ap.add_argument("--repeticoes", type=int, default=5, help="cv-repetida: quantas sementes de dobras")
    ap.add_argument("--zip", default=str(ROCOLE_ZIP), help="zip do RoCoLe (Kaggle), com as fotos originais")
    ap.add_argument("--param1", type=int, default=rh.PARAM1)
    ap.add_argument("--param2", type=int, default=rh.PARAM2)
    args = ap.parse_args(argv)
    conj = carregar_conjunto(args.dataset, args.zip)
    conj.pasta.mkdir(parents=True, exist_ok=True)
    if args.etapa in ("mascaras", "extrair") and conj.zip is not None and not os.path.isfile(conj.zip):
        raise SystemExit(f"zip nao encontrado: {conj.zip}. Coloque o RoCoLe em data/archive.zip ou passe --zip CAMINHO")
    if args.etapa == "mascaras":
        if conj.nome == "rocole":
            gerar_mascaras_baixa_confianca(conj.meta, conj.zip)
        else:
            gerar_mascaras_do_conjunto(conj)
    elif args.etapa == "extrair":
        cache = conj.cache(args.param1, args.param2)
        print(f"{conj.nome}: extraindo circulos e atributos de {len(conj.meta)} fotos (param1 {args.param1}, param2 "
              f"{args.param2})...", flush=True)
        dados = atributos_do_conjunto(conj, args.param1, args.param2)
        print(f"cache: {cache}")
        print("\ncirculos por classe e lado (so o TREINO):")
        print(resumo_propostas(conj.meta, dados).round(3).to_string())
        if (args.param1, args.param2) == (rh.PARAM1, rh.PARAM2):
            figura_propostas(conj, conj.pasta / "propostas_treino.jpg")
            print(f"figura: {conj.pasta / 'propostas_treino.jpg'}")
    elif args.etapa == "cv":
        rodar_cv(conj)
    elif args.etapa == "final":
        ajustar_final(conj)
    elif args.etapa == "fusao":
        if conj.nome != "rocole":
            raise SystemExit("a fusao com a ResNet-50 so existe para o RoCoLe (nao ha rede treinada por dobra nos outros)")
        rodar_fusao(conj.meta)
    elif args.etapa == "cv-repetida":
        rodar_cv_repetida(conj, args.repeticoes)


if __name__ == "__main__":
    main()
