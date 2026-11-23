"""Valida o RoCoLe (zip do Kaggle), remove copias, gera cache 512x288 e divide 70/15/15 por planta.

Uso: python -P src/preparar_rocole.py [--zip data/archive.zip]

Regras (fixadas antes de rodar; ver a secao Dados de docs/RELATORIO_ROCOLE.md):
- copias exatas (SHA-256): fica uma; se os rotulos divergem, saem todas;
- quase-copias (pHash <= 4) com rotulos diferentes saem todas;
- grupos de divisao, montados sobre TODAS as imagens antes de descartar qualquer uma: mesma planta (C<n>P<m>,
  inclui os dois lados H/E de cada folha), mesma copia exata ou pHash <= 10 (refotos da mesma folha).
  Um grupo nunca atravessa splits.
"""
import argparse
import hashlib
import io
import re
import warnings
import zipfile
from pathlib import Path, PurePosixPath

import imagehash
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from comum import CLASSES_ROCOLE, OUT_ROCOLE, ROCOLE_ALTURA, ROCOLE_CACHE, ROCOLE_LARGURA, ROCOLE_ZIP, SEED

LIMIAR_PHASH = 4  # quase-copia: rotulos diferentes => descarta
LIMIAR_GRUPO = 10  # refoto da mesma folha: so junta no mesmo grupo (nao descarta)
EXTENSOES = (".jpg", ".jpeg", ".png")
PREFIXO_CLASSE = "coffee___"


def listar_imagens(nomes):
    """Entradas do zip do tipo coffee___<classe>/<arquivo> com extensao de imagem (qualquer caixa)."""
    saida = []
    for n in nomes:
        partes = PurePosixPath(n).parts
        if len(partes) == 2 and partes[0].startswith(PREFIXO_CLASSE) and n.lower().endswith(EXTENSOES):
            saida.append(n)
    return saida


def verificar_entrada(nomes):
    """Para cedo, com mensagem clara: zip sem coffee___<classe>/<arquivo>, classe desconhecida ou id repetido
    (dois arquivos com o mesmo nome se sobrescreveriam no cache)."""
    if not nomes:
        raise SystemExit("nenhuma imagem coffee___<classe>/<arquivo> no zip: as pastas das classes devem estar na raiz "
                         "do zip (o zip tem uma pasta a mais?)")
    info = [ler_nome(n) for n in nomes]
    desconhecidas = sorted({i["classe"] for i in info} - set(CLASSES_ROCOLE))
    if desconhecidas:
        raise SystemExit(f"classes desconhecidas no zip: {desconhecidas}; esperadas: {CLASSES_ROCOLE}")
    ids = pd.Series([i["id"] for i in info])
    repetidos = sorted(ids[ids.duplicated()].unique())
    if repetidos:
        raise SystemExit(f"nomes de arquivo repetidos em pastas diferentes: {repetidos[:10]}")


def ler_nome(arquivo):
    """'coffee___rust/C10P10E2.jpg' -> id C10P10E2, classe rust, planta C10P10, lado E (H=face, E=verso)."""
    pasta, nome = PurePosixPath(arquivo).parts
    id_ = PurePosixPath(nome).stem
    m = re.match(r"C(\d+)P(\d+)", id_)
    lado = re.fullmatch(r"C\d+P\d+([HE])\d+", id_)
    return {"id": id_, "classe": pasta[len(PREFIXO_CLASSE):],
            "planta": f"C{m[1]}P{m[2]}" if m else id_, "lado": lado[1] if lado else ""}


def resolver_copias_exatas(df):
    """Mesmo SHA-256: fica a primeira por id; se as classes divergem, saem todas."""
    descartadas, manter = [], []
    for _, g in df.groupby("sha256", sort=False):
        g = g.sort_values("id")
        if g["classe"].nunique() > 1:
            descartadas += [(a, "copia exata com rotulo conflitante") for a in g["arquivo"]]
            continue
        manter.append(g.index[0])
        descartadas += [(a, f"copia exata de {g['id'].iloc[0]}") for a in g["arquivo"].iloc[1:]]
    return df.loc[sorted(manter)].reset_index(drop=True), descartadas


def pares_proximos(hashes, limiar=LIMIAR_PHASH):
    """Pares (i, j), i < j, com distancia de Hamming entre os pHash <= limiar."""
    return [(i, j) for i in range(len(hashes)) for j in range(i + 1, len(hashes))
            if hashes[i] - hashes[j] <= limiar]


def componentes(n, pares):
    """Componentes conexas (union-find); o rotulo e o menor indice da componente."""
    pai = list(range(n))

    def raiz(i):
        while pai[i] != i:
            pai[i] = pai[pai[i]]
            i = pai[i]
        return i

    for a, b in pares:
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            pai[max(ra, rb)] = min(ra, rb)
    return [raiz(i) for i in range(n)]


def remover_quase_copias_conflitantes(df):
    """Remove inteira toda componente de quase-copias (coluna 'quase') com mais de uma classe."""
    conflito = df.groupby("quase")["classe"].transform("nunique") > 1
    descartadas = [(a, "quase-copia com rotulo conflitante") for a in df.loc[conflito, "arquivo"]]
    return df[~conflito].reset_index(drop=True), descartadas


def agrupar(df, pares=()):
    """Grupo de divisao: une linhas da mesma planta, do mesmo SHA-256 e os pares (i, j) dados (quase-copias).
    Deve rodar sobre todas as imagens, antes dos descartes, para nao perder ligacoes."""
    pares, primeiro = list(pares), {}
    for col in ("planta", "sha256"):
        if col not in df:
            continue
        for i, v in enumerate(df[col]):
            chave = (col, v)
            if chave in primeiro:
                pares.append((primeiro[chave], i))
            else:
                primeiro[chave] = i
    return componentes(len(df), pares)


def filtrar_e_agrupar(todos):
    """Todas as imagens decodificadas (id, arquivo, classe, planta, sha256, phash) -> (mantidas com 'grupo',
    descartadas, numero de pares de quase-copias <= LIMIAR_PHASH entre as restantes)."""
    todos = todos.reset_index(drop=True)
    todos["grupo"] = agrupar(todos, pares_proximos(todos["phash"].tolist(), LIMIAR_GRUPO))
    df, descartadas = resolver_copias_exatas(todos)
    pares = pares_proximos(df["phash"].tolist())
    df["quase"] = componentes(len(df), pares)
    df, d = remover_quase_copias_conflitantes(df)
    return df, descartadas + d, len(pares)


def dividir(df, seed=SEED):
    """70/15/15 estratificado por classe e agrupado por 'grupo' (mesma receita de preparar_dados.py)."""
    split = pd.Series("", index=df.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
        resto_idx, teste_idx = next(sgkf.split(df, df["classe"], df["grupo"]))
        split.iloc[teste_idx] = "test"
        resto = df.iloc[resto_idx]
        sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
        tr, va = next(sgkf2.split(resto, resto["classe"], resto["grupo"]))
        split.loc[resto.index[tr]] = "train"
        split.loc[resto.index[va]] = "val"
    return split


def verificar(df):
    """Nenhum grupo e nenhum SHA-256 em mais de um split; com a coluna 'phash', nenhum par com pHash <= LIMIAR_GRUPO
    em splits diferentes (checagem independente dos grupos)."""
    assert (df.groupby("grupo")["split"].nunique() == 1).all(), "grupo dividido entre splits!"
    assert (df.groupby("sha256")["split"].nunique() == 1).all(), "mesma imagem em mais de um split!"
    if "phash" in df:
        s = df["split"].to_numpy()
        cruzados = [(i, j) for i, j in pares_proximos(df["phash"].tolist(), LIMIAR_GRUPO) if s[i] != s[j]]
        assert not cruzados, f"{len(cruzados)} pares com pHash <= {LIMIAR_GRUPO} em splits diferentes!"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", type=Path, default=ROCOLE_ZIP, help="zip do RoCoLe (Kaggle)")
    args = ap.parse_args(argv)
    if not args.zip.is_file():
        raise SystemExit(f"zip nao encontrado: {args.zip}. "
                         "Coloque o RoCoLe em data/archive.zip ou passe --zip CAMINHO")
    OUT_ROCOLE.mkdir(parents=True, exist_ok=True)
    ROCOLE_CACHE.mkdir(parents=True, exist_ok=True)

    try:
        zf = zipfile.ZipFile(args.zip)
    except zipfile.BadZipFile:
        raise SystemExit(f"{args.zip} nao e um zip valido")
    with zf as z:
        arquivos = [n for n in z.namelist() if not n.endswith("/")]
        nomes = listar_imagens(arquivos)
        verificar_entrada(nomes)
        if len(nomes) < len(arquivos):
            print(f"aviso: {len(arquivos) - len(nomes)} arquivos do zip nao sao imagens de coffee___<classe>/ "
                  "e foram ignorados")

        # 1a passada: valida, SHA-256 e pHash (a imagem inteira nao fica na memoria)
        linhas, descartadas = [], []
        for n in nomes:
            dados = z.read(n)
            try:
                with Image.open(io.BytesIO(dados)) as im:
                    im.load()
                    ph = imagehash.phash(im.convert("RGB"))
            except Exception as e:  # imagem corrompida
                descartadas.append((n, f"corrompida: {str(e)[:50]}"))
                continue
            info = ler_nome(n)
            if info["planta"] == info["id"]:
                print(f"aviso: nome fora do padrao C<n>P<m>: {n} (vira um grupo proprio)")
            linhas.append({**info, "arquivo": n, "sha256": hashlib.sha256(dados).hexdigest(), "phash": ph})
        n_lidas = len(linhas) + len(descartadas)

        # grupos sobre todas as imagens, depois copias exatas e quase-copias conflitantes
        df, d, n_pares = filtrar_e_agrupar(pd.DataFrame(linhas))
        descartadas += d

        # divisao
        df["split"] = dividir(df)
        verificar(df)

        # 2a passada: cache 512x288 so das imagens mantidas
        for r in df.itertuples():
            with Image.open(io.BytesIO(z.read(r.arquivo))) as im:
                im.convert("RGB").resize((ROCOLE_LARGURA, ROCOLE_ALTURA), Image.LANCZOS).save(
                    ROCOLE_CACHE / f"{r.id}.jpg", quality=95)
    faltando = [i for i in df["id"] if not (ROCOLE_CACHE / f"{i}.jpg").exists()]
    assert not faltando, f"{len(faltando)} imagens sem cache: {faltando[:5]}"

    desc = pd.DataFrame(descartadas, columns=["arquivo", "motivo"])
    motivo = desc["motivo"].str.replace(r"^copia exata de .*", "copia exata (fica uma)", regex=True)
    print(f"imagens lidas: {n_lidas} | mantidas: {len(df)} | descartadas: {len(desc)}")
    print(motivo.value_counts().to_string())
    print(f"pares de quase-copias (pHash <= {LIMIAR_PHASH}) entre as restantes: {n_pares}")
    print(f"grupos (planta + copia exata + pHash <= {LIMIAR_GRUPO}, sobre todas as imagens): "
          f"{df['grupo'].nunique()} | maior: {df.groupby('grupo').size().max()} imagens")
    print(f"\ngrupos, SHA-256 e pares com pHash <= {LIMIAR_GRUPO} atravessando splits: 0 (verificado)\n")
    print(pd.crosstab(df["split"], df["classe"], margins=True).loc[["train", "val", "test", "All"]])

    colunas = ["id", "arquivo", "classe", "planta", "lado", "grupo", "sha256", "split"]
    df[colunas].to_csv(OUT_ROCOLE / "metadados.csv", index=False)
    desc.to_csv(OUT_ROCOLE / "descartadas.csv", index=False)
    print(f"\ngravado: {OUT_ROCOLE / 'metadados.csv'}, {OUT_ROCOLE / 'descartadas.csv'}, cache em {ROCOLE_CACHE}")


if __name__ == "__main__":
    main()
