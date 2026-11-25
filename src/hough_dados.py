"""O conjunto de dados do modelo de Hough: o BRACOL.

Os comandos de treino e teste pegam daqui as fotos, a classe de cada uma, a divisao em treino/validacao/teste, as
mascaras da folha e onde gravar resultados e modelo.

BRACOL: cafe arabica, folha destacada sobre fundo claro (data/bracol_raw); a divisao e a de
outputs/hough_bracol/metadados.csv, agrupada por pHash. Classes: saudavel ou doenca (qualquer uma das quatro; nenhuma
doenca vale mais que outra), com a saudavel no indice 0. As dobras da validacao cruzada sao estratificadas pela
combinacao das doencas.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from comum import CACHE_HOUGH, DADOS, MODELOS, SAIDAS

CONJUNTOS = ("bracol",)
CLASSES_BRACOL = ["healthy", "disease"]  # saudavel ou doenca
BRACOL_IMAGENS = DADOS / "bracol_raw" / "coffee-datasets" / "coffee-datasets" / "leaf" / "images"
BRACOL_META = SAIDAS / "hough_bracol" / "metadados.csv"
DOENCAS_BRACOL = ["miner", "rust", "phoma", "cercospora"]
MASCARAS_BRACOL = CACHE_HOUGH / "mascaras_bracol"
ARTEFATO_HOUGH = MODELOS / "bracol_hough.joblib"  # o modelo de Hough (passo 1 do aplicativo)


def classe_bracol(df):
    """'disease' se a folha tem qualquer uma das quatro doencas; senao 'healthy'."""
    return (df[DOENCAS_BRACOL].sum(axis=1) > 0).map({True: "disease", False: "healthy"})


def estrato_bracol(df):
    """A combinacao das doencas da folha, como texto ('0101' = ferrugem e cercosporiose): estratifica as dobras."""
    return df[DOENCAS_BRACOL].astype(int).astype(str).agg("".join, axis=1)


def meta_bracol():
    if not BRACOL_META.exists():
        raise SystemExit(f"{BRACOL_META} nao existe: e o arquivo com a divisao do BRACOL")
    m = pd.read_csv(BRACOL_META)
    m["id"] = m["id"].astype(str)
    return m.assign(arquivo=[str(BRACOL_IMAGENS / f"{i}.jpg") for i in m["id"]], classe=classe_bracol(m),
                    estrato=estrato_bracol(m), origem="bracol").rename(columns={"severity": "severidade"})


@dataclass
class Conjunto:
    nome: str
    meta: pd.DataFrame  # id, arquivo, classe, grupo, split, origem e estrato
    classes: list  # nomes das classes, na ordem dos indices (0 = saudavel)
    pasta: Path  # onde ficam os resultados
    artefato: Path  # o modelo gravado
    pastas_mascaras: list
    arquivo_confiancas: Path  # csv com id e confianca do SAM
    arquivos_meta: list  # os csv que definem a divisao
    positiva: str = "disease"  # a classe que os resumos dos circulos acompanham

    def cache(self, param1, param2):
        """Arquivo .joblib com os circulos e atributos das fotos."""
        return CACHE_HOUGH / f"{self.nome}_atributos_p{param1}_{param2}.joblib"

    def sha256(self):
        """Impressao digital da divisao: o modelo guarda a sua, e o teste para se ela mudou."""
        h = hashlib.sha256()
        for arq in self.arquivos_meta:
            h.update(Path(arq).read_bytes())
        return h.hexdigest()


def carregar_conjunto(nome="bracol"):
    if nome != "bracol":
        raise SystemExit(f"conjunto desconhecido: '{nome}'. Use: {', '.join(CONJUNTOS)}")
    pasta = SAIDAS / "hough_bracol"
    return Conjunto("bracol", meta_bracol(), list(CLASSES_BRACOL), pasta, ARTEFATO_HOUGH, [MASCARAS_BRACOL],
                    pasta / "folhas_sam.csv", [BRACOL_META])


def adicionar_opcao_dataset(ap):
    """--dataset bracol e o atalho --bracol (argparse), os dos comandos de uso. So ha o BRACOL, que e o padrao."""
    ap.add_argument("--dataset", choices=CONJUNTOS, default="bracol", help="conjunto de dados (padrao: %(default)s)")
    ap.add_argument("--bracol", dest="dataset", action="store_const", const="bracol", help="o mesmo que --dataset bracol")
