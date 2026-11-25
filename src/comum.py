"""Caminhos e constantes. Saidas e pesos ficam em outputs/ e models/; as fotos e os caches ficam em data/, na raiz do
repositorio (fora do git). A variavel de ambiente CAFE_DADOS troca essa pasta (util quando os dados ficam em outro
lugar)."""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent  # raiz do repositorio
DADOS = Path(os.environ["CAFE_DADOS"]) if os.environ.get("CAFE_DADOS") else RAIZ / "data"
SAIDAS = RAIZ / "outputs"
MODELOS = RAIZ / "models"

CACHE_HOUGH = DADOS / "cache_hough"  # mascaras do SAM e atributos dos circulos do Hough (hough_treinar.py)

BRACOL_CACHE = DADOS / "cache_512x256"  # fotos do BRACOL reduzidas para a ConvNeXt-Tiny (convnext_bracol.py)
BRACOL_LARGURA, BRACOL_ALTURA = 512, 256  # originais 2048x1024 (2:1)

# passo 1 do aplicativo: "incerto" quando nem saudavel nem doenca passa disto (app.py, hough_avaliar.py)
LIMITE_INCERTO = 0.60

SEED = 42
MEDIA = (0.485, 0.456, 0.406)  # normalizacao do ImageNet
DESVIO = (0.229, 0.224, 0.225)
