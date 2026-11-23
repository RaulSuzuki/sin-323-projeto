"""Caminhos e constantes do RoCoLe. Saidas e pesos ficam em outputs/ e models/; o zip e os caches ficam em data/, na raiz do
repositorio, ao lado dos dados do BRACOL (fora do git). A variavel de ambiente ROCOLE_DADOS troca essa pasta
(util quando os dados ficam em outro lugar)."""
import os
from pathlib import Path

PASTA_ROCOLE = Path(__file__).resolve().parent.parent  # raiz do repositorio
RAIZ = PASTA_ROCOLE
DADOS = Path(os.environ["ROCOLE_DADOS"]) if os.environ.get("ROCOLE_DADOS") else RAIZ / "data"
OUT_ROCOLE = PASTA_ROCOLE / "outputs"
MODELOS_ROCOLE = PASTA_ROCOLE / "models"

ROCOLE_ZIP = DADOS / "archive.zip"  # nome do arquivo do Kaggle; --zip troca
ROCOLE_CACHE = DADOS / "cache_rocole_512x288"
ROCOLE_CACHE_RECORTE = DADOS / "cache_rocole_recorte_512x288"  # folha-alvo recortada (folha_sam.py)
ROCOLE_CACHE_RECORTE_MASCARA = DADOS / "cache_rocole_recorte_mascara_512x288"  # idem, fundo apagado
ROCOLE_MASCARAS = DADOS / "mascaras_rocole"  # mascaras do SAM, 960x540
ROCOLE_CACHE_HOUGH = DADOS / "cache_rocole_hough"  # atributos dos circulos do Hough (hough_treinar.py)

CLASSES_ROCOLE = ["healthy", "red_spider_mite", "rust"]  # ordem alfabetica, como o ImageFolder
NOMES_ROCOLE_PT = {"healthy": "Saudavel", "red_spider_mite": "Acaro-vermelho", "rust": "Ferrugem"}
ROCOLE_LARGURA, ROCOLE_ALTURA = 512, 288  # originais 1920x1080 (16:9)

SEED = 42  # as mesmas do src/comum.py do BRACOL
MEDIA = (0.485, 0.456, 0.406)  # normalizacao do ImageNet
DESVIO = (0.229, 0.224, 0.225)
