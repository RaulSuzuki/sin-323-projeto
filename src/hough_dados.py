"""Conjuntos de dados do detector de ferrugem com o Hough: rocole ou bracol.

Os comandos de treino e teste recebem --dataset (ou os atalhos --rocole, --bracol) e pegam daqui tudo o que
muda de um conjunto para outro: as fotos, a classe de cada uma, a divisao em treino/validacao/teste, as mascaras da
folha e onde gravar resultados e modelo.

- rocole: cafe robusta fotografado na planta (zip do Kaggle); classes saudavel, acaro-vermelho, ferrugem.
- bracol: cafe arabica, folha destacada sobre fundo claro (data/bracol_raw); a divisao e a do modelo da raiz
  (outputs/hough_bracol/metadados.csv, agrupada por pHash). Classes: saudavel, outra doenca SEM ferrugem, ferrugem (com ou sem
  outra doenca junto).

Cada conjunto tem o seu modelo, treinado e testado nele mesmo. Nos dois, a classe de indice 2 e a ferrugem e a de
indice 0 e a folha saudavel.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from comum import (CLASSES_ROCOLE, DADOS, MODELOS_ROCOLE, OUT_ROCOLE, ROCOLE_CACHE_HOUGH, ROCOLE_MASCARAS,
                          ROCOLE_ZIP)

CONJUNTOS = ("rocole", "bracol")
CLASSES_GERAIS = ["healthy", "other", "rust"]  # saudavel, outra lesao (sem ferrugem), ferrugem
BRACOL_IMAGENS = DADOS / "bracol_raw" / "coffee-datasets" / "coffee-datasets" / "leaf" / "images"
BRACOL_META = OUT_ROCOLE / "hough_bracol" / "metadados.csv"
DOENCAS_BRACOL = ["miner", "rust", "phoma", "cercospora"]
MASCARAS_BAIXA = ROCOLE_CACHE_HOUGH / "mascaras_baixa_confianca"  # RoCoLe: as que folha_sam.py recusou por confianca
MASCARAS_BRACOL = ROCOLE_CACHE_HOUGH / "mascaras_bracol"


def classe_bracol(df):
    """'rust' se a folha tem ferrugem (com ou sem outra doenca); 'other' se so tem outra doenca; senao 'healthy'."""
    outras = df[[d for d in DOENCAS_BRACOL if d != "rust"]].sum(axis=1) > 0
    return (df["rust"] == 1).map({True: "rust", False: ""}).where(df["rust"] == 1, outras.map({True: "other", False: "healthy"}))


def meta_rocole():
    arq = OUT_ROCOLE / "metadados.csv"
    if not arq.exists():
        raise SystemExit(f"{arq} nao existe: rode preparar_rocole.py")
    return pd.read_csv(arq).assign(origem="rocole")


def meta_bracol():
    if not BRACOL_META.exists():
        raise SystemExit(f"{BRACOL_META} nao existe: e o arquivo com a divisao do BRACOL")
    m = pd.read_csv(BRACOL_META)
    m["id"] = m["id"].astype(str)
    return m.assign(arquivo=[str(BRACOL_IMAGENS / f"{i}.jpg") for i in m["id"]], classe=classe_bracol(m), lado="",
                    origem="bracol").rename(columns={"severity": "severidade"})


@dataclass
class Conjunto:
    nome: str
    meta: pd.DataFrame  # id, arquivo, classe, grupo, split, lado, origem
    classes: list  # nomes das classes, na ordem dos indices (0 saudavel, 1 outra lesao, 2 ferrugem)
    pasta: Path  # onde ficam os resultados
    artefato: Path  # o modelo gravado
    pastas_mascaras: list
    arquivo_confiancas: Path  # csv com id e confianca do SAM
    arquivos_meta: list  # os csv que definem a divisao
    zip: Path = None  # so o RoCoLe le as fotos de um zip; o BRACOL le de arquivos soltos

    def cache(self, param1, param2):
        """Arquivo .joblib com os circulos e atributos das fotos."""
        prefixo = "" if self.nome == "rocole" else f"{self.nome}_"
        return ROCOLE_CACHE_HOUGH / f"{prefixo}atributos_p{param1}_{param2}.joblib"

    def sha256(self):
        """Impressao digital da divisao: o modelo guarda a sua, e o teste para se ela mudou."""
        h = hashlib.sha256()
        for arq in self.arquivos_meta:
            h.update(Path(arq).read_bytes())
        return h.hexdigest()


def carregar_conjunto(nome, caminho_zip=None):
    caminho_zip = Path(caminho_zip) if caminho_zip else ROCOLE_ZIP
    if nome == "rocole":
        return Conjunto("rocole", meta_rocole(), list(CLASSES_ROCOLE), OUT_ROCOLE / "hough",
                        MODELOS_ROCOLE / "rocole_hough.joblib", [ROCOLE_MASCARAS, MASCARAS_BAIXA],
                        OUT_ROCOLE / "folhas_sam.csv", [OUT_ROCOLE / "metadados.csv"], zip=caminho_zip)
    if nome == "bracol":
        pasta = OUT_ROCOLE / "hough_bracol"
        return Conjunto("bracol", meta_bracol(), list(CLASSES_GERAIS), pasta, MODELOS_ROCOLE / "bracol_hough.joblib",
                        [MASCARAS_BRACOL], pasta / "folhas_sam.csv", [BRACOL_META])
    raise SystemExit(f"conjunto desconhecido: '{nome}'. Use um de: {', '.join(CONJUNTOS)}")


def adicionar_opcao_dataset(ap, padrao="rocole"):
    """--dataset rocole|bracol e os atalhos --rocole, --bracol (argparse)."""
    ap.add_argument("--dataset", choices=CONJUNTOS, default=padrao, help="conjunto de dados (padrao: %(default)s)")
    for nome in CONJUNTOS:
        ap.add_argument(f"--{nome}", dest="dataset", action="store_const", const=nome, help=f"o mesmo que --dataset {nome}")
