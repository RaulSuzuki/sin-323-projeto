"""Pontua o modelo de Hough de um conjunto no teste do MESMO conjunto, UMA vez por modelo final.

Uso:
  python -P src/rocole_hough_avaliar.py            # RoCoLe: so o modelo de Hough
  python -P src/rocole_hough_avaliar.py --fusao    # RoCoLe: fusao tardia com a ResNet-50 (se adotada em `treinar.py fusao`)
  python -P src/rocole_hough_avaliar.py --bracol   # BRACOL: modelo treinado no BRACOL, no teste do BRACOL
  python -P src/rocole_hough_avaliar.py --folhas   # RoCoLe: F1 de ferrugem pela folha, das predicoes ja gravadas
  python -P src/rocole_hough_avaliar.py --conferencia-externa  # modelo do RoCoLe (fusao, SAM na hora) em 598 folhas do BRACOL

--dataset (ou --rocole, --bracol) escolhe o conjunto: o modelo treinado nele (models/<conjunto>_hough.joblib) e
pontuado no teste dele (RoCoLe -> RoCoLe, BRACOL -> BRACOL).

No teste do RoCoLe (208 fotos): F1, sensibilidade, especificidade e acuracia balanceada de ferrugem, com IC 95% por
bootstrap de plantas, ao lado do detector por regras e da ResNet-50 nas mesmas fotos (predicoes ja gravadas); as 3
classes; ferrugem por lado da folha; e a folha inteira (face + verso). Configuracao, peso da fusao e limiares vem so
de treino + validacao. Para se o teste ja foi pontuado para aquele modelo (--refazer passa por cima, de proposito).

--conferencia-externa nao escolhe nada: roda o modelo do RoCoLe pronto, com o SAM achando a folha na hora, em folhas de
outra especie e outro tipo de foto, para ver o que esperar dele fora do RoCoLe.
"""
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rocole_hough as rh
import rocole_hough_treinar as ht
from rocole_comum import NOMES_ROCOLE_PT, OUT_ROCOLE, ROCOLE_ZIP, SEED
from rocole_hough_dados import BRACOL_IMAGENS, BRACOL_META, DOENCAS_BRACOL, adicionar_opcao_dataset, carregar_conjunto
from rocole_regras import bootstrap_binario, metricas_binarias, taxas_por_classe_e_lado

PASTA = ht.PASTA
RESULTADOS = PASTA / "resultados_teste.json"
CLASSES = ht.CLASSES
RUST = ht.RUST


def sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def conferir_primeira_vez(arq=RESULTADOS, refazer=False):
    """O teste e pontuado uma vez por modelo final: para se o resultado ja existe."""
    if Path(arq).exists() and not refazer:
        raise SystemExit(f"{arq} ja existe: o teste ja foi pontuado para este modelo. Use --refazer so de proposito "
                         "(por exemplo, depois de treinar de novo).")


def conferir_divisao(modelo, metadados=OUT_ROCOLE / "metadados.csv"):
    """Para se o modelo foi treinado com outra divisao: o teste atual poderia ter fotos do treino dele."""
    if modelo.get("metadados_sha256") != sha256(metadados):
        raise SystemExit("o modelo foi treinado com outra divisao (metadados.csv mudou): rode rocole_hough_treinar.py "
                         "cv e final de novo")


def indice_predito(p, limiar):
    """Indice da classe predita: o da ferrugem se P(ferrugem) >= limiar; senao, o da mais provavel entre as outras."""
    p = np.asarray(p, float)
    outras = p.copy()
    outras[:, RUST] = -1.0
    return np.where(p[:, RUST] >= limiar, RUST, outras.argmax(1))


def classe_predita(p, limiar, nomes=None):
    """'rust' se P(ferrugem) >= limiar; senao, a mais provavel entre as outras classes (nomes: os do conjunto)."""
    return np.array(CLASSES if nomes is None else nomes)[indice_predito(p, limiar)]


def modelos_comparados(fusao):
    """(nome, coluna booleana 'chamou de ferrugem') dos modelos da tabela do teste; o primeiro e o pontuado agora."""
    base = [("hough", "hough_ferrugem"), ("regras", "regras_ferrugem"), ("resnet50", "resnet50_ferrugem")]
    return ([("fusao", "fusao_ferrugem")] if fusao else []) + base


def wilson(k, n, z=1.96):
    """IC 95% de Wilson para a proporcao k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centro, meia = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centro - meia) / (1 + z * z / n)), min(1.0, (centro + meia) / (1 + z * z / n))


def grupos_bracol(meta, n_outras=150, seed=SEED):
    """Folhas do BRACOL para a conferencia externa: so ferrugem, saudaveis e uma amostra das que tem outra doenca
    sem ferrugem. Folhas com ferrugem E outra doenca ficam de fora (o rotulo nao diria o que o modelo viu)."""
    soma = meta[DOENCAS_BRACOL].sum(axis=1)
    outras = meta[(meta["rust"] == 0) & (soma > 0)]
    return {"so_ferrugem": meta[(meta["rust"] == 1) & (soma == 1)], "saudavel": meta[soma == 0],
            "outras_doencas_sem_ferrugem": outras.sample(min(n_outras, len(outras)), random_state=seed).sort_values("id")}


def f1_por_folha(df):
    """Ferrugem pela folha (face + verso; existe se aparece em algum lado), a partir de predicoes por foto (id,
    classe, classe_predita): folhas, acertos (vp), perdas (fn), alarmes (fp) e F1 para cada estrategia de
    rocole_validacao (so a face, so o verso, as duas fotos)."""
    from rocole_validacao import ESTRATEGIAS, presenca_folhas
    pres = presenca_folhas(df)
    r = pres[pres["doenca"] == "rust"]
    v = r["verdade"].to_numpy(bool)
    linhas = []
    for est in ESTRATEGIAS:
        p = r[est].to_numpy(bool)
        vp, fn, fp = int((v & p).sum()), int((v & ~p).sum()), int((~v & p).sum())
        linhas.append({"estrategia": est, "folhas": int(len(v)), "folhas_com_ferrugem": int(v.sum()), "vp": vp, "fn": fn,
                       "fp": fp, "f1": 2 * vp / (2 * vp + fp + fn) if vp else 0.0})
    return pd.DataFrame(linhas)


def resumir_folhas():
    """Le as predicoes de teste JA gravadas (nao roda modelo nenhum) e grava folhas_teste_f1.csv: F1 de ferrugem
    pela folha para a fusao, o modelo de Hough e a ResNet-50."""
    fontes = [("fusao", PASTA / "predicoes_teste_fusao.csv"), ("hough", PASTA / "predicoes_teste.csv"),
              ("resnet50", OUT_ROCOLE / "resnet50" / "predicoes_teste.csv")]
    partes = []
    for nome, arq in fontes:
        if arq.exists():
            df = pd.read_csv(arq).rename(columns={"classe_real": "classe"})
            partes.append(f1_por_folha(df[["id", "classe", "classe_predita"]]).assign(modelo=nome))
    tab = pd.concat(partes, ignore_index=True)[["modelo", "estrategia", "folhas", "folhas_com_ferrugem", "vp", "fn", "fp", "f1"]]
    tab.to_csv(PASTA / "folhas_teste_f1.csv", index=False)
    print("--- Ferrugem pela folha inteira no teste (das predicoes ja gravadas) ---")
    print(tab.round(4).to_string(index=False))


def figura_exemplos(te, modelo, caminho_zip, caminho, coluna="hough_ferrugem", nome="Modelo de Hough", limiar=None,
                    por_tipo=2):
    """Acertos e erros sorteados, com o contorno da folha e os circulos do Hough coloridos pela pontuacao de
    ferrugem do nivel 1 (vermelho = parece ferrugem, azul = nao parece)."""
    rng = np.random.default_rng(SEED)
    y, pred = (te["classe"] == "rust"), te[coluna].astype(bool)
    tipos = [("Ferrugem encontrada", y & pred), ("Ferrugem perdida", y & ~pred),
             ("Saudavel com alarme", (te["classe"] == "healthy") & pred),
             ("Acaro com alarme", (te["classe"] == "red_spider_mite") & pred),
             ("Sem ferrugem, sem alarme", ~y & ~pred)]
    linhas = []
    for rotulo, filtro in tipos:
        cand = te[filtro]
        for i in rng.choice(len(cand), size=min(por_tipo, len(cand)), replace=False) if len(cand) else []:
            linhas.append((rotulo, cand.iloc[int(i)]))
    z = zipfile.ZipFile(caminho_zip)
    conf = ht.confiancas_sam()
    n = len(linhas)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(15, 3.6 * ((n + 1) // 2)), layout="constrained", squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (rotulo, r) in zip(axes.ravel(), linhas):
        img = cv2.imdecode(np.frombuffer(z.read(r["arquivo"]), np.uint8), cv2.IMREAD_COLOR)
        e = rh.extrair(img, ht.carregar_mascara(r["id"], img.shape[:2]), conf.get(r["id"], 0.0), modelo["param1"],
                       modelo["param2"])
        vis = e["folha"].img.copy()
        cont, _ = cv2.findContours(e["folha"].mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cont, -1, (255, 255, 255), 2)
        if len(e["circulos"]) and modelo["nivel1"] is not None:
            p = modelo["nivel1"].predict_proba(e["at_circulos"])[:, list(modelo["nivel1"].classes_).index(RUST)]
            for (x, yy, raio), pr in sorted(zip(e["circulos"], p), key=lambda t: t[1]):
                cv2.circle(vis, (int(x), int(yy)), int(raio) + 2, (int(255 * (1 - pr)), 0, int(255 * pr)), 3)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        acerto = bool(r[coluna]) == (r["classe"] == "rust")
        ax.set_title(f"{rotulo}\n{r['id']} | {NOMES_ROCOLE_PT[r['classe']]} | P(ferrugem) = {r['p_rust']:.2f} | "
                     f"{len(e['circulos'])} circulos", fontsize=10, color="green" if acerto else "red")
    limiar = modelo["limiar"] if limiar is None else limiar
    fig.suptitle(f"{nome} no teste (limiar {limiar:.2f}; circulo vermelho = parece ferrugem)")
    fig.savefig(caminho, dpi=80)
    plt.close(fig)


def pontuar_teste(caminho_zip, fusao=False, refazer=False):
    """Pontua no teste o modelo de Hough (fusao=False) ou a fusao com a ResNet-50 (fusao=True). Os arquivos da
    fusao levam o sufixo _fusao; os do modelo de Hough sozinho nao sao tocados."""
    sufixo = "_fusao" if fusao else ""
    arq_resultados = PASTA / f"resultados_teste{sufixo}.json"
    conferir_primeira_vez(arq_resultados, refazer)
    if not ht.ARTEFATO.exists():
        raise SystemExit(f"{ht.ARTEFATO} nao existe: rode rocole_hough_treinar.py cv e final")
    if not os.path.isfile(caminho_zip):
        raise SystemExit(f"zip nao encontrado: {caminho_zip}. Coloque o RoCoLe em data/archive.zip ou passe --zip CAMINHO")
    arq_ref = OUT_ROCOLE / "regras" / "predicoes_teste.csv"
    arq_cnn = OUT_ROCOLE / "resnet50" / "predicoes_teste.csv"
    for arq in (arq_ref, arq_cnn):
        if not arq.exists():
            raise SystemExit(f"{arq} nao existe: rode rocole_avaliar.py e rocole_regras.py testar")
    from rocole_avaliar import bootstrap_por_grupo, metricas  # importa o torch; so aqui
    from rocole_validacao import avaliar_folhas, folhas_com_ic

    modelo = joblib.load(ht.ARTEFATO)
    conferir_divisao(modelo)
    if fusao and not modelo.get("fusao"):
        raise SystemExit("o modelo gravado nao tem fusao: rode rocole_hough_treinar.py fusao (ela so entra se passar "
                         "a regra na validacao cruzada)")
    meta = pd.read_csv(OUT_ROCOLE / "metadados.csv")
    te = meta[meta["split"] == "test"].reset_index(drop=True)
    dados = ht.extrair_tudo(meta, caminho_zip, modelo["param1"], modelo["param2"],
                            cache=ht.arquivo_cache(modelo["param1"], modelo["param2"]))
    p_hough = ht.prever(modelo, te["id"], dados)
    te["hough_ferrugem"] = p_hough[:, RUST] >= modelo["limiar"]
    if fusao:  # probabilidades da ResNet-50 final no teste, ja gravadas por rocole_avaliar.py
        cnn = pd.read_csv(arq_cnn).set_index("id")
        assert te["id"].isin(cnn.index).all(), "as predicoes da ResNet-50 nao cobrem o teste atual"
        p_cnn = cnn.loc[te["id"], [f"p_{c}" for c in CLASSES]].to_numpy()
        p, limiar = ht.fundir(p_hough, p_cnn, modelo["fusao"]["w"]), modelo["fusao"]["limiar"]
        te["p_hough_rust"], te["p_resnet50_rust"] = p_hough[:, RUST], p_cnn[:, RUST]
    else:
        p, limiar = p_hough, modelo["limiar"]
    for j, c in enumerate(CLASSES):
        te[f"p_{c}"] = p[:, j]
    te["classe_predita"] = classe_predita(p, limiar)
    modelos = modelos_comparados(fusao)
    nome, coluna = modelos[0]
    te[coluna] = te["classe_predita"] == "rust"
    te["n_circulos"] = [len(dados[i]["circulos"]) for i in te["id"]]
    ref = pd.read_csv(arq_ref).set_index("id")
    assert te["id"].isin(ref.index).all(), "as predicoes de regras/ResNet-50 nao cobrem o teste atual"
    te["regras_ferrugem"] = te["id"].map(ref["regras_ferrugem"]).astype(bool)
    te["resnet50_ferrugem"] = te["id"].map(ref["resnet50_ferrugem"]).astype(bool)
    colunas = (["id", "classe", "lado", "grupo", "n_circulos"] + [f"p_{c}" for c in CLASSES]
               + (["p_hough_rust", "p_resnet50_rust"] if fusao else []) + ["classe_predita"] + [c for _, c in modelos])
    te[colunas].to_csv(PASTA / f"predicoes_teste{sufixo}.csv", index=False)

    y, grupos = (te["classe"] == "rust").to_numpy(), te["grupo"].to_numpy()
    resultados = {"modelo": nome, "n_teste": int(len(te)), "n_ferrugem": int(y.sum()), "limiar": limiar,
                  "param1": modelo["param1"], "param2": modelo["param2"], "variante": modelo["variante"],
                  "fusao": modelo["fusao"] if fusao else None}
    ics = []
    for n, col in modelos:
        pred = te[col].to_numpy()
        resultados[n] = {**metricas_binarias(y, pred),
                         "confusao": {"vp": int((y & pred).sum()), "fn": int((y & ~pred).sum()),
                                      "fp": int((~y & pred).sum()), "vn": int((~y & ~pred).sum())},
                         "taxa_ferrugem_por_classe": {c: float(pred[(te["classe"] == c).to_numpy()].mean()) for c in CLASSES}}
        lados = taxas_por_classe_e_lado(te, col)
        resultados[n]["taxa_ferrugem_por_classe_e_lado"] = {c: lados.loc[c].dropna().to_dict() for c in lados.index}
        ics.append(bootstrap_binario(y, pred, grupos).assign(modelo=n))
    ic = pd.concat(ics).reset_index(names="metrica").set_index(["modelo", "metrica"])

    y3 = te["classe"].map(CLASSES.index).to_numpy()
    pred3 = te["classe_predita"].map(CLASSES.index).to_numpy()
    confusao3 = pd.crosstab(te["classe"], te["classe_predita"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    resultados[nome]["tres_classes"] = metricas(y3, pred3)
    resultados[nome]["matriz_confusao_3_classes"] = confusao3.to_numpy().tolist()
    ic3 = bootstrap_por_grupo(y3, pred3, grupos).assign(modelo=f"{nome}_3_classes")
    ic = pd.concat([ic, ic3.reset_index(names="metrica").set_index(["modelo", "metrica"])])
    ic.to_csv(PASTA / f"teste_intervalos_confianca{sufixo}.csv")

    folhas = avaliar_folhas(te[["id", "classe", "classe_predita"]])
    folhas.to_csv(PASTA / f"folhas_teste{sufixo}.csv", index=False)
    folhas_com_ic(te[["id", "classe", "classe_predita"]], meta).to_csv(PASTA / f"folhas_teste{sufixo}_ic.csv", index=False)
    with open(arq_resultados, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)

    print(f"--- Teste, {nome} ({len(te)} fotos, {int(y.sum())} com ferrugem; limiar {limiar:.2f}; IC 95% por bootstrap "
          f"de {te['grupo'].nunique()} grupos de plantas) ---")
    print(ic.round(4).to_string())
    print("\nfracao das fotos de cada classe chamada de 'ferrugem':")
    print(pd.DataFrame({n: resultados[n]["taxa_ferrugem_por_classe"] for n, _ in modelos}).round(3).to_string())
    print("\npor lado da folha (H = face, E = verso):")
    print(pd.concat({n: taxas_por_classe_e_lado(te, c) for n, c in modelos}, axis=1).round(3).to_string())
    print(f"\n3 classes, {nome} (real nas linhas, predito nas colunas):")
    print(confusao3.to_string())
    print(f"\nfolha inteira (face + verso), {nome}:")
    print(folhas.round(3).to_string(index=False))
    figura_exemplos(te, modelo, caminho_zip, PASTA / f"exemplos_teste{sufixo}.png", coluna,
                    "Fusao Hough + ResNet-50" if fusao else "Modelo de Hough", limiar)
    print(f"\nResultados salvos em: {PASTA}")


def pontuar_em(conj, refazer=False):
    """Pontua, no TESTE do conjunto `conj`, o modelo de Hough treinado nele (conj.artefato). Grava
    resultados_teste.json, predicoes_teste.csv e teste_intervalos_confianca.csv na pasta de `conj`, uma vez."""
    arq = conj.pasta / "resultados_teste.json"
    conferir_primeira_vez(arq, refazer)
    if not conj.artefato.exists():
        raise SystemExit(f"{conj.artefato} nao existe: rode rocole_hough_treinar.py cv e final --dataset {conj.nome}")
    modelo = joblib.load(conj.artefato)
    if modelo.get("metadados_sha256") != conj.sha256():
        raise SystemExit(f"o modelo de '{conj.nome}' foi treinado com outra divisao: rode rocole_hough_treinar.py cv e "
                         f"final --dataset {conj.nome} de novo")
    te = conj.meta[conj.meta["split"] == "test"].reset_index(drop=True)
    dados = ht.atributos_do_conjunto(conj, modelo["param1"], modelo["param2"])
    p = ht.prever(modelo, te["id"], dados)
    limiar = modelo["limiar"]
    nomes_modelo = list(modelo.get("classes", CLASSES))
    y3 = te["classe"].map(list(conj.classes).index).to_numpy()
    pred3 = indice_predito(p, limiar)
    y, pred, grupos = y3 == RUST, pred3 == RUST, te["grupo"].to_numpy()
    confusao = np.zeros((3, 3), int)
    np.add.at(confusao, (y3, pred3), 1)
    ic = bootstrap_binario(y, pred, grupos)
    resultados = {"conjunto": conj.nome, "n_teste": int(len(te)), "n_ferrugem": int(y.sum()),
                  "limiar": limiar, "variante": modelo["variante"], "classes": list(conj.classes),
                  "ferrugem": {**metricas_binarias(y, pred), "vp": int((y & pred).sum()), "fn": int((y & ~pred).sum()),
                               "fp": int((~y & pred).sum()), "vn": int((~y & ~pred).sum())},
                  "ic95": {m: [float(ic.loc[m, "IC95_inf"]), float(ic.loc[m, "IC95_sup"])] for m in ic.index},
                  "taxa_ferrugem_por_classe": {c: float(pred[y3 == j].mean()) for j, c in enumerate(conj.classes)},
                  "matriz_confusao": confusao.tolist()}
    conj.pasta.mkdir(parents=True, exist_ok=True)
    te.assign(**{f"p_{c}": p[:, j] for j, c in enumerate(nomes_modelo)}, ferrugem=pred)[
        ["id", "classe", "grupo"] + [f"p_{c}" for c in nomes_modelo] + ["ferrugem"]].to_csv(
        conj.pasta / "predicoes_teste.csv", index=False)
    ic.to_csv(conj.pasta / "teste_intervalos_confianca.csv")
    with open(arq, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2)
    r = resultados["ferrugem"]
    print(f"--- Teste de '{conj.nome}' ({len(te)} fotos, {int(y.sum())} com ferrugem) com o modelo de Hough treinado em "
          f"'{conj.nome}' (limiar {limiar:.2f}) ---")
    print(ic.round(4).to_string())
    print(f"ferrugem achada: {r['vp']} de {r['vp'] + r['fn']} | alarmes: {r['fp']} em {r['fp'] + r['vn']}")
    print("fracao chamada de 'ferrugem' por classe do teste:",
          {c: round(v, 3) for c, v in resultados["taxa_ferrugem_por_classe"].items()})
    print("classes (real nas linhas, predito nas colunas: " + ", ".join(conj.classes) + "):")
    print(confusao)
    return resultados


def conferir_bracol(n_outras=150):
    """Conferencia externa no BRACOL (cafe arabica, folha destacada sobre fundo claro): roda o modelo pronto, com o
    SAM achando a folha na hora, e conta quantas folhas ele chama de ferrugem em cada grupo. Nada e escolhido aqui."""
    from rocole_hough_inferir import Diagnosticador, desenhar
    if not BRACOL_META.exists() or not BRACOL_IMAGENS.exists():
        raise SystemExit(f"{BRACOL_META} ou {BRACOL_IMAGENS} nao existe: prepare o BRACOL (README da raiz)")
    grupos = grupos_bracol(pd.read_csv(BRACOL_META), n_outras)
    diag = Diagnosticador()
    m = diag.modelo
    linhas, exemplos = [], []
    for grupo, tab in grupos.items():
        print(f"{grupo}: {len(tab)} folhas...", flush=True)
        for k, r in enumerate(tab.itertuples()):
            img = cv2.imread(str(BRACOL_IMAGENS / f"{r.id}.jpg"), cv2.IMREAD_COLOR)
            if img is None:
                continue
            d = diag.diagnosticar(img)
            pr = d["p_resnet"]
            linhas.append({"id": r.id, "grupo": grupo, "severidade": r.severity, "ferrugem": d["ferrugem"],
                           "p_rust": d["p"]["rust"], "p_hough_rust": d["p_hough"]["rust"],
                           "p_resnet50_rust": pr["rust"] if pr else np.nan,
                           "so_hough_ferrugem": d["p_hough"]["rust"] >= m["limiar"],
                           "so_resnet50_ferrugem": bool(pr and max(pr, key=pr.get) == "rust"),
                           "n_circulos": len(d["circulos"]), "confianca_sam": d["confianca"], "aviso": d["aviso"]})
            if k < 4 and grupo != "outras_doencas_sem_ferrugem":
                vis = desenhar(img, d)
                vis = cv2.resize(vis, (640, 320), interpolation=cv2.INTER_AREA)
                cv2.putText(vis, f"{r.id} {grupo}: {'FERRUGEM' if d['ferrugem'] else 'sem ferrugem'} "
                                 f"p={d['p']['rust']:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                exemplos.append(vis)
    res = pd.DataFrame(linhas)
    res.to_csv(PASTA / "bracol_externo.csv", index=False)
    resumo = []
    for grupo, g in res.groupby("grupo", sort=False):
        for nome, col in [("modelo final", "ferrugem"), ("so Hough", "so_hough_ferrugem"), ("so ResNet-50", "so_resnet50_ferrugem")]:
            k, n = int(g[col].sum()), len(g)
            inf, sup = wilson(k, n)
            resumo.append({"grupo": grupo, "modelo": nome, "folhas": n, "chamadas_de_ferrugem": k, "fracao": k / n,
                           "IC95_inf": inf, "IC95_sup": sup})
    resumo = pd.DataFrame(resumo)
    resumo.to_csv(PASTA / "bracol_resumo.csv", index=False)
    if exemplos:
        cv2.imwrite(str(PASTA / "bracol_exemplos.jpg"), np.vstack([np.hstack(exemplos[i:i + 2]) for i in range(0, len(exemplos) - 1, 2)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("\n--- Conferencia externa no BRACOL (arabica, fundo claro). NADA foi ajustado nestas fotos. ---")
    print(resumo.round(3).to_string(index=False))
    sev = res[res["grupo"] == "so_ferrugem"].groupby("severidade")["ferrugem"].agg(["mean", "sum", "count"])
    print("\nso ferrugem, por severidade do BRACOL (fracao chamada de ferrugem pelo modelo final):")
    print(sev.round(3).to_string())
    print(f"\ncontorno pelo SAM em {(res['confianca_sam'] > 0).mean():.1%} das folhas | mediana de circulos: "
          f"{res['n_circulos'].median():.0f}")
    print(f"gravado: {PASTA / 'bracol_externo.csv'}, bracol_resumo.csv, bracol_exemplos.jpg")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    adicionar_opcao_dataset(ap)
    ap.add_argument("--zip", default=str(ROCOLE_ZIP))
    ap.add_argument("--fusao", action="store_true", help="RoCoLe: pontua a fusao com a ResNet-50 (arquivos com sufixo _fusao)")
    ap.add_argument("--conferencia-externa", action="store_true",
                    help="modelo do RoCoLe (fusao, SAM na hora) em folhas do BRACOL, em vez do teste")
    ap.add_argument("--folhas", action="store_true", help="RoCoLe: F1 de ferrugem pela folha, das predicoes ja gravadas")
    ap.add_argument("--refazer", action="store_true", help="pontua o teste de novo, de proposito")
    args = ap.parse_args(argv)
    if args.folhas:
        resumir_folhas()
    elif args.conferencia_externa:
        conferir_bracol()
    elif args.dataset == "rocole":
        pontuar_teste(args.zip, fusao=args.fusao, refazer=args.refazer)  # o teste completo do RoCoLe
    else:
        if args.fusao:
            raise SystemExit("--fusao so vale para o RoCoLe")
        pontuar_em(carregar_conjunto(args.dataset, args.zip), refazer=args.refazer)


if __name__ == "__main__":
    main()
