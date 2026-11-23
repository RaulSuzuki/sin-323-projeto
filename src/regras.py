"""Detector de ferrugem por regras no RoCoLe, replica de Rodriguez-Gallo, Escobar-Benitez e Rodriguez-Lainez,
"Robust Coffee Rust Detection Using UAV-Based Aerial RGB Imagery", AgriEngineering 5 (2023) 1415-1431.

Uso:
  python -P src/regras.py ajustar [--zip data/archive.zip]  # escolhe parametros no treino, confere na validacao
  python -P src/regras.py testar  [--zip data/archive.zip]  # pontua o teste uma vez e compara com a ResNet-50

Pergunta binaria: ha ferrugem visivel na foto? Positivo = pasta rust; negativo = healthy e red_spider_mite.
Pipeline (como no artigo): variancia local por canal (janela circular 5x5, como o filtro Variance do ImageJ) ->
mascara de vermelho/laranja (HSV) -> contornos preenchidos -> abertura e fechamento (elipse 5x5) -> Hough de
circulos -> "ferrugem" se houver pelo menos k circulos. O artigo nao publica os limiares de cor, os parametros do
Hough nem a regra de decisao: aqui eles sao escolhidos no treino, numa grade pequena.
"""
import argparse
import json
import os
import zipfile
from multiprocessing import Pool

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comum import NOMES_ROCOLE_PT, OUT_ROCOLE, ROCOLE_MASCARAS, ROCOLE_ZIP, SEED

RAIO_VARIANCIA = 2  # janela circular 5x5 (N = 5 no artigo; raio 2 no ImageJ)
# nome: (matiz maxima, matiz minima do vermelho que da a volta, saturacao minima, brilho minimo); matiz do OpenCV 0-179
PRESETS_COR = {
    "estreito": (15, 170, 120, 120),
    "medio": (20, 165, 90, 90),
    "largo": (25, 160, 60, 60),
}
PARAM2_GRADE = (12, 18, 25)  # limiar do acumulador do Hough: menor = mais circulos
K_GRADE = (1, 2, 3, 5, 8, 12)  # minimo de circulos para dizer "ferrugem"
RAIO_MIN, RAIO_MAX = 4, 60  # raio das lesoes em pixels, nas fotos 1920x1080
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
CHAVES = ["sensibilidade", "especificidade", "acuracia_balanceada", "f1", "acuracia"]
PASTA = OUT_ROCOLE / "regras"
PARAMS = PASTA / "parametros.json"
PASTA_FOLHA = OUT_ROCOLE / "regras_folha"  # mesmo detector, so dentro da mascara da folha-alvo (SAM)


def _disco(raio):
    yy, xx = np.mgrid[-raio:raio + 1, -raio:raio + 1]
    k = (xx ** 2 + yy ** 2 <= raio ** 2 + 1).astype(np.float32)  # mesmo formato do kernel circular do ImageJ
    return k / k.sum()


def variancia_local(img, raio=RAIO_VARIANCIA):
    """Variancia local de cada canal numa janela circular, cortada em 0-255 (como o ImageJ faz em imagens RGB)."""
    k = _disco(raio)
    f = img.astype(np.float32)
    media = cv2.filter2D(f, -1, k, borderType=cv2.BORDER_REFLECT)
    media2 = cv2.filter2D(f * f, -1, k, borderType=cv2.BORDER_REFLECT)
    return np.clip(np.rint(media2 - media * media), 0, 255).astype(np.uint8)


def mascara_cor(var_bgr, preset):
    """Pixels vermelhos/laranja do mapa de variancia (bordas das lesoes de ferrugem). Devolve 0/255."""
    hmax, hmin2, smin, vmin = PRESETS_COR[preset]
    h, s, v = cv2.split(cv2.cvtColor(var_bgr, cv2.COLOR_BGR2HSV))
    m = ((h <= hmax) | (h >= hmin2)) & (s >= smin) & (v >= vmin)
    return m.astype(np.uint8) * 255


def limpar_mascara(mascara):
    """Contornos externos preenchidos, depois abertura e fechamento com elipse 5x5 (nesta ordem, como no artigo)."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cheia = np.zeros_like(mascara)
    cv2.drawContours(cheia, contornos, -1, 255, thickness=cv2.FILLED)
    cheia = cv2.morphologyEx(cheia, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(cheia, cv2.MORPH_CLOSE, KERNEL)


def circulos(mascara_limpa, param2):
    """Transformada de Hough de circulos na mascara limpa. Devolve (N, 3) inteiros: x, y, raio."""
    borrada = cv2.GaussianBlur(mascara_limpa, (9, 9), 2)
    c = cv2.HoughCircles(borrada, cv2.HOUGH_GRADIENT, dp=1, minDist=2 * RAIO_MIN, param1=100, param2=param2,
                         minRadius=RAIO_MIN, maxRadius=RAIO_MAX)
    return np.zeros((0, 3), int) if c is None else np.rint(c[0, :, :3]).astype(int)


def contar_grade(img_bgr, presets=PRESETS_COR, param2s=PARAM2_GRADE, mascara=None):
    """Numero de circulos para cada combinacao de preset de cor e sensibilidade do Hough."""
    var = variancia_local(img_bgr)
    if mascara is not None:  # so a folha-alvo: fora dela a variancia vira 0 (nada vermelho/laranja)
        var[~mascara] = 0
    saida = {}
    for nome in presets:
        m = limpar_mascara(mascara_cor(var, nome))
        for p2 in param2s:
            saida[f"{nome}_p{p2}"] = int(len(circulos(m, p2)))
    return saida


def metricas_binarias(y, pred):
    """Sensibilidade, especificidade, acuracia balanceada, F1 e precisao da classe positiva (ferrugem), e acuracia.
    Razoes sem denominador viram NaN (o bootstrap as ignora)."""
    y, p = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, tn = int((y & p).sum()), int((~y & ~p).sum())
    fp, fn = int((~y & p).sum()), int((y & ~p).sum())
    div = lambda a, b: a / b if b else float("nan")
    sens, espec = div(tp, tp + fn), div(tn, tn + fp)
    return {"sensibilidade": sens, "especificidade": espec, "acuracia_balanceada": (sens + espec) / 2,
            "f1": div(2 * tp, 2 * tp + fp + fn), "precisao": div(tp, tp + fp), "acuracia": div(tp + tn, len(y))}


def bootstrap_binario(y, pred, grupos, b=2000, seed=SEED):
    """IC 95% reamostrando grupos (plantas) com reposicao, como em cnn_avaliar.py."""
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


def escolher_parametros(contagens, y, ks=K_GRADE):
    """Combinacao (preset, param2) e k com a maior acuracia balanceada. Empate: fica a primeira, na ordem da grade."""
    melhor = None
    for col in contagens.columns:
        for k in ks:
            m = metricas_binarias(y, contagens[col].to_numpy() >= k)
            if melhor is None or m["acuracia_balanceada"] > melhor["acuracia_balanceada"] + 1e-12:
                preset, p2 = col.rsplit("_p", 1)
                melhor = {"preset": preset, "param2": int(p2), "k": int(k), **m}
    return melhor


def carregar_parametros(caminho=PARAMS):
    if not caminho.exists():
        raise SystemExit(f"{caminho} nao existe: rode regras.py ajustar")
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- leitura em paralelo das fotos originais do zip
_ZIP = None


def _abrir_zip(caminho):
    global _ZIP
    cv2.setNumThreads(1)  # um processo por nucleo; sem threads extras do OpenCV
    _ZIP = zipfile.ZipFile(caminho)


def _ler(arquivo, z=None):
    return cv2.imdecode(np.frombuffer((z or _ZIP).read(arquivo), np.uint8), cv2.IMREAD_COLOR)


def _ler_mascara(caminho, forma):
    m = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (forma[1], forma[0]), interpolation=cv2.INTER_NEAREST) > 0


def _contar(tarefa):
    arquivo, caminho_mascara = tarefa
    img = _ler(arquivo)
    return contar_grade(img, mascara=_ler_mascara(caminho_mascara, img.shape[:2]) if caminho_mascara else None)


def _detectar(tarefa):
    arquivo, preset, p2 = tarefa
    return int(len(circulos(limpar_mascara(mascara_cor(variancia_local(_ler(arquivo)), preset)), p2)))


def _mapa_paralelo(func, itens, caminho_zip):
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_abrir_zip, initargs=(str(caminho_zip),)) as pool:
        return pool.map(func, itens, chunksize=4)


def _taxas_por_classe(classes, pred):
    """Fracao das fotos de cada classe em que o detector diz 'ferrugem'."""
    s = pd.Series(np.asarray(pred).astype(bool), index=np.asarray(classes))
    return {c: float(s[s.index == c].mean()) for c in ["healthy", "red_spider_mite", "rust"]}


def taxas_por_classe_e_lado(df, coluna):
    """Fracao com 'ferrugem' por classe (linhas) e lado da folha (colunas: H = face, E = verso)."""
    return df.pivot_table(index="classe", columns="lado", values=coluna, aggfunc="mean")


def ajustar(meta, caminho_zip, com_mascara=False):
    """Escolhe parametros no treino e confere na validacao. com_mascara: so dentro da folha-alvo (fotos sem mascara
    aceita entram inteiras, como na CNN); saidas em outputs/regras_folha/."""
    pasta = PASTA_FOLHA if com_mascara else PASTA
    params_arq = pasta / "parametros.json"
    sub = meta[meta["split"].isin(["train", "val"])].reset_index(drop=True)
    mascaras = [ROCOLE_MASCARAS / f"{i}.png" for i in sub["id"]]
    tarefas = [(a, str(m) if com_mascara and m.exists() else None) for a, m in zip(sub["arquivo"], mascaras)]
    if com_mascara:
        print(f"com mascara da folha: {sum(t[1] is not None for t in tarefas)} de {len(tarefas)} fotos", flush=True)
    print(f"contando circulos em {len(sub)} fotos (treino + validacao), {len(PRESETS_COR) * len(PARAM2_GRADE)} "
          "combinacoes por foto...", flush=True)
    cont = pd.DataFrame(_mapa_paralelo(_contar, tarefas, caminho_zip))
    cont = pd.concat([sub[["id", "classe", "grupo", "split"]], cont], axis=1)
    cont.to_csv(pasta / "contagens_ajuste.csv", index=False)
    colunas = [f"{n}_p{p}" for n in PRESETS_COR for p in PARAM2_GRADE]

    tr, va = cont[cont["split"] == "train"], cont[cont["split"] == "val"]
    y_tr, y_va = (tr["classe"] == "rust").to_numpy(), (va["classe"] == "rust").to_numpy()
    grade = pd.DataFrame([{"combinacao": c, "k": k, **metricas_binarias(y_tr, tr[c].to_numpy() >= k)}
                          for c in colunas for k in K_GRADE])
    grade.to_csv(pasta / "ajuste_grade_treino.csv", index=False)
    p = escolher_parametros(tr[colunas], y_tr)
    col = f"{p['preset']}_p{p['param2']}"
    pred_va = va[col].to_numpy() >= p["k"]
    m_va = metricas_binarias(y_va, pred_va)

    saida = {"preset": p["preset"], "param2": p["param2"], "k": p["k"], "cor": PRESETS_COR[p["preset"]],
             "raio_variancia": RAIO_VARIANCIA, "raio_min": RAIO_MIN, "raio_max": RAIO_MAX,
             "treino": {c: p[c] for c in CHAVES + ["precisao"]},
             "treino_taxa_ferrugem_por_classe": _taxas_por_classe(tr["classe"], tr[col].to_numpy() >= p["k"]),
             "val": m_va, "val_taxa_ferrugem_por_classe": _taxas_por_classe(va["classe"], pred_va)}
    with open(params_arq, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2)

    print("\nmelhores combinacoes no treino (acuracia balanceada):")
    print(grade.sort_values("acuracia_balanceada", ascending=False).head(8).round(4).to_string(index=False))
    print(f"\nescolhido: cor '{p['preset']}', param2 {p['param2']}, k {p['k']}")
    print(f"treino ({len(tr)}): acuracia balanceada {p['acuracia_balanceada']:.4f} | sensibilidade "
          f"{p['sensibilidade']:.4f} | especificidade {p['especificidade']:.4f}")
    print(f"validacao ({len(va)}): acuracia balanceada {m_va['acuracia_balanceada']:.4f} | sensibilidade "
          f"{m_va['sensibilidade']:.4f} | especificidade {m_va['especificidade']:.4f}")
    print("validacao, fracao com 'ferrugem' por classe:", {k: round(v, 3) for k, v in saida["val_taxa_ferrugem_por_classe"].items()})
    print(f"\ngravado: {params_arq}")
    if com_mascara and PARAMS.exists():
        sem = carregar_parametros(PARAMS)
        print(f"comparacao na validacao, acuracia balanceada: foto inteira {sem['val']['acuracia_balanceada']:.4f} | "
              f"so a folha {m_va['acuracia_balanceada']:.4f}")


def _desenhar(img_bgr, circ, cor=(0, 0, 255)):
    out = img_bgr.copy()
    for x, y, r in circ:
        cv2.circle(out, (int(x), int(y)), int(r) + 6, cor, 5)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def figura_etapas(z, arquivo, id_, params, caminho):
    img = _ler(arquivo, z)
    var = variancia_local(img)
    m = mascara_cor(var, params["preset"])
    limpa = limpar_mascara(m)
    circ = circulos(limpa, params["param2"])
    paineis = [("Original", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), ("Variancia local (5x5)", cv2.cvtColor(var, cv2.COLOR_BGR2RGB)),
               ("Mascara vermelho/laranja", m), ("Contornos + abertura/fechamento", limpa),
               (f"Hough: {len(circ)} circulos", _desenhar(img, circ))]
    fig, axes = plt.subplots(1, 5, figsize=(22, 3.4), layout="constrained")
    for ax, (t, im) in zip(axes, paineis):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Etapas do detector por regras - {id_} (teste)")
    fig.savefig(caminho, dpi=100)
    plt.close(fig)


def figura_exemplos(z, linhas, params, caminho):
    n = len(linhas)
    fig, axes = plt.subplots((n + 3) // 4, 4, figsize=(18, 2.9 * ((n + 3) // 4)), layout="constrained", squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (rotulo, r) in zip(axes.ravel(), linhas):
        img = _ler(r.arquivo, z)
        circ = circulos(limpar_mascara(mascara_cor(variancia_local(img), params["preset"])), params["param2"])
        ax.imshow(_desenhar(img, circ))
        acerto = bool(r.regras_ferrugem) == (r.classe == "rust")
        ax.set_title(f"{rotulo}\n{r.id} | {NOMES_ROCOLE_PT[r.classe]} | {len(circ)} circulos", fontsize=9,
                     color="green" if acerto else "red")
    fig.suptitle(f"Detector por regras no teste (cor '{params['preset']}', param2 {params['param2']}, k {params['k']})")
    fig.savefig(caminho, dpi=90)
    plt.close(fig)


def testar(meta, caminho_zip):
    params = carregar_parametros()
    arq_resnet = OUT_ROCOLE / "resnet50" / "predicoes_teste.csv"
    if not arq_resnet.exists():
        raise SystemExit(f"{arq_resnet} nao existe: rode cnn_avaliar.py antes (comparacao com a ResNet-50)")
    te = meta[meta["split"] == "test"].reset_index(drop=True)
    print(f"detectando em {len(te)} fotos de teste com cor '{params['preset']}', param2 {params['param2']}, "
          f"k {params['k']}...", flush=True)
    tarefas = [(a, params["preset"], params["param2"]) for a in te["arquivo"]]
    te["n_circulos"] = _mapa_paralelo(_detectar, tarefas, caminho_zip)
    te["regras_ferrugem"] = te["n_circulos"] >= params["k"]
    resnet = pd.read_csv(arq_resnet).set_index("id")["classe_predita"]
    te["resnet50_ferrugem"] = te["id"].map(resnet).eq("rust")
    assert te["id"].isin(resnet.index).all(), "predicoes da ResNet-50 nao cobrem o teste atual: rode cnn_avaliar.py"
    te[["id", "classe", "grupo", "n_circulos", "regras_ferrugem", "resnet50_ferrugem"]].to_csv(
        PASTA / "predicoes_teste.csv", index=False)

    y = (te["classe"] == "rust").to_numpy()
    resultados, ics = {"parametros": {k: params[k] for k in ("preset", "param2", "k")}, "n_teste": len(te)}, []
    for nome, col in [("regras", "regras_ferrugem"), ("resnet50", "resnet50_ferrugem")]:
        pred = te[col].to_numpy()
        resultados[nome] = {**metricas_binarias(y, pred), "taxa_ferrugem_por_classe": _taxas_por_classe(te["classe"], pred)}
        ics.append(bootstrap_binario(y, pred, te["grupo"].to_numpy()).assign(modelo=nome))
    ic = pd.concat(ics).reset_index(names="metrica").set_index(["modelo", "metrica"])
    ic.to_csv(PASTA / "teste_intervalos_confianca.csv")
    with open(PASTA / "resultados_teste.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)

    print(f"\n--- Teste ({len(te)} fotos, {int(y.sum())} com ferrugem; IC 95% por bootstrap de "
          f"{te['grupo'].nunique()} grupos de plantas) ---")
    print(ic.round(4).to_string())
    print("\nfracao das fotos de cada classe chamada de 'ferrugem':")
    print(pd.DataFrame({n: resultados[n]["taxa_ferrugem_por_classe"] for n in ("regras", "resnet50")}).round(3).to_string())
    lados = {n: taxas_por_classe_e_lado(te, col) for n, col in [("regras", "regras_ferrugem"), ("resnet50", "resnet50_ferrugem")]}
    for n, t in lados.items():
        resultados[n]["taxa_ferrugem_por_classe_e_lado"] = {c: t.loc[c].dropna().to_dict() for c in t.index}
    with open(PASTA / "resultados_teste.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)
    print("\npor lado da folha (H = face, E = verso):")
    print(pd.concat(lados, axis=1).round(3).to_string())

    # figuras (poucas fotos, no processo principal)
    z = zipfile.ZipFile(caminho_zip)
    vp = te[te["regras_ferrugem"] & y]
    exemplo = vp.sort_values("n_circulos").iloc[len(vp) // 2] if len(vp) else te.iloc[0]
    figura_etapas(z, exemplo["arquivo"], exemplo["id"], params, PASTA / "etapas_detector.png")
    rng = np.random.default_rng(SEED)
    linhas = []
    for rotulo, filtro in [("Ferrugem encontrada", te["regras_ferrugem"] & y), ("Ferrugem perdida", ~te["regras_ferrugem"] & y),
                           ("Saudavel com alarme", te["regras_ferrugem"] & (te["classe"] == "healthy")),
                           ("Acaro com alarme", te["regras_ferrugem"] & (te["classe"] == "red_spider_mite")),
                           ("Acaro sem alarme", ~te["regras_ferrugem"] & (te["classe"] == "red_spider_mite"))]:
        cand = te[filtro]
        for i in rng.choice(len(cand), size=min(2, len(cand)), replace=False) if len(cand) else []:
            linhas.append((rotulo, cand.iloc[int(i)]))
    figura_exemplos(z, linhas, params, PASTA / "exemplos_detector.png")
    print(f"\nResultados salvos em: {PASTA}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("etapa", choices=["ajustar", "testar"])
    ap.add_argument("--zip", default=str(ROCOLE_ZIP), help="zip do RoCoLe (Kaggle), com as fotos originais 1920x1080")
    ap.add_argument("--mascara", action="store_true", help="so dentro da folha-alvo (mascaras de folha_sam.py)")
    args = ap.parse_args(argv)
    if args.mascara and args.etapa == "testar":
        raise SystemExit("--mascara so vale para 'ajustar': o teste fica para o modelo final")
    if args.mascara and not ROCOLE_MASCARAS.exists():
        raise SystemExit(f"{ROCOLE_MASCARAS} nao existe: rode folha_sam.py")
    if not os.path.isfile(args.zip):
        raise SystemExit(f"zip nao encontrado: {args.zip}. Coloque o RoCoLe em data/archive.zip ou passe --zip CAMINHO")
    meta_csv = OUT_ROCOLE / "metadados.csv"
    if not meta_csv.exists():
        raise SystemExit(f"{meta_csv} nao existe: rode preparar_rocole.py")
    PASTA.mkdir(parents=True, exist_ok=True)
    PASTA_FOLHA.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(meta_csv)
    if args.etapa == "ajustar":
        ajustar(meta, args.zip, com_mascara=args.mascara)
    else:
        testar(meta, args.zip)


if __name__ == "__main__":
    main()
