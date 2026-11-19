"""Dataset, transformacoes e modelos (ResNet-50 ou ConvNeXt-Tiny) do RoCoLe."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms as T

from rocole_comum import (CLASSES_ROCOLE, DESVIO, MEDIA, MODELOS_ROCOLE, OUT_ROCOLE, ROCOLE_ALTURA, ROCOLE_CACHE,
                   ROCOLE_CACHE_RECORTE, ROCOLE_CACHE_RECORTE_MASCARA, ROCOLE_LARGURA)

MODELOS_SUPORTADOS = ("resnet50", "convnext_tiny")
# entrada da CNN -> cache 512x288: foto inteira, folha-alvo recortada, folha recortada com fundo apagado
ENTRADAS = {"foto": ROCOLE_CACHE, "recorte": ROCOLE_CACHE_RECORTE, "recorte_mascara": ROCOLE_CACHE_RECORTE_MASCARA}


def nome_execucao(modelo, entrada="foto"):
    """Nome dos pesos e da pasta de saida: 'resnet50' para a foto inteira, 'resnet50_recorte' etc. para as outras."""
    return modelo if entrada == "foto" else f"{modelo}_{entrada}"


def transformacao(treino):
    """Augmentation do main.py (SIN 393), adaptada a entrada 16:9 de 512x288. Validacao/teste: so normaliza."""
    base = [T.ToTensor(), T.Normalize(MEDIA, DESVIO)]
    if not treino:
        return T.Compose(base)
    return T.Compose([
        T.RandomResizedCrop((ROCOLE_ALTURA, ROCOLE_LARGURA), scale=(0.8, 1.0), ratio=(1.6, 1.95)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        *base,
    ])


class FolhasRocole(Dataset):
    """Imagens de um split (train, val ou test) de outputs/metadados.csv, lidas do cache 512x288.
    Devolve (tensor 3x288x512, indice da classe em CLASSES_ROCOLE, id)."""

    def __init__(self, split, treino=False, meta=None, cache=ROCOLE_CACHE):
        if meta is None:
            arq = OUT_ROCOLE / "metadados.csv"
            if not arq.exists() or not cache.exists():
                raise SystemExit(f"{arq} ou {cache} nao existe: rode rocole_preparar.py")
            meta = pd.read_csv(arq)
        self.meta = meta[meta["split"] == split].reset_index(drop=True)
        self.cache = cache
        self.tf = transformacao(treino)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]
        with Image.open(self.cache / f"{r['id']}.jpg") as im:
            img = im.convert("RGB")
        return self.tf(img), CLASSES_ROCOLE.index(r["classe"]), str(r["id"])


def construir_modelo(nome, n_classes, pretreinado=True):
    """ResNet-50 (pesos ImageNet V2) ou ConvNeXt-Tiny (ImageNet V1) com a ultima camada trocada."""
    if nome == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretreinado else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif nome == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretreinado else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    else:
        raise ValueError(f"modelo '{nome}' nao suportado; use {' ou '.join(MODELOS_SUPORTADOS)}")
    return m


def camada_alvo(modelo, nome):
    """Ultima etapa convolucional, usada pelo Grad-CAM."""
    return modelo.layer4 if nome.startswith("resnet") else modelo.features[-1]


def congelar_backbone(modelo, nome, congelar):
    """Congelado: so a cabeca treina. Descongelado: a rede inteira treina (como freeze_backbone do main.py)."""
    prefixo = "fc." if nome == "resnet50" else "classifier."
    for n, p in modelo.named_parameters():
        p.requires_grad = (not congelar) or n.startswith(prefixo)


def caminho_pesos(nome):
    return MODELOS_ROCOLE / f"rocole_{nome}.pt"


def _sha256(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def registrar_divisao(pesos, metadados=OUT_ROCOLE / "metadados.csv"):
    """Grava, ao lado do checkpoint (<pesos>.json), o SHA-256 do metadados.csv usado no treino."""
    with open(Path(pesos).with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"metadados_sha256": _sha256(metadados)}, f, indent=2)


def verificar_divisao(pesos, metadados=OUT_ROCOLE / "metadados.csv"):
    """Para se o checkpoint foi treinado com outra divisao: avalia-lo no teste atual misturaria treino e teste."""
    registro = Path(pesos).with_suffix(".json")
    if not registro.exists():
        raise SystemExit(f"{registro} nao existe (sem registro da divisao usada no treino): rode rocole_treinar.py")
    with open(registro, encoding="utf-8") as f:
        if json.load(f)["metadados_sha256"] != _sha256(metadados):
            raise SystemExit(f"{pesos} foi treinado com outra divisao (metadados.csv mudou): rode rocole_treinar.py")


def carregar(nome, disp, caminho=None, entrada="foto"):
    """Modelo treinado por rocole_treinar.py, em eval() no dispositivo disp. Confere a divisao do treino."""
    caminho = caminho or caminho_pesos(nome_execucao(nome, entrada))
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} nao existe: rode rocole_treinar.py --modelo {nome}")
    verificar_divisao(caminho)
    m = construir_modelo(nome, len(CLASSES_ROCOLE), pretreinado=False)
    m.load_state_dict(torch.load(caminho, map_location=disp, weights_only=True))
    return m.to(disp).eval()
