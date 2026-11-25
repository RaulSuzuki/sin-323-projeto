"""Aplica os modelos a uma foto qualquer de folha, SEM mascara: o contorno da folha e calculado aqui (SAM com um
ponto no centro; se o SAM nao achar a folha, pela cor). Usado pelo aplicativo (app/app.py).

    diag = Diagnosticador(artefato=..., doencas=...)  # SAM e ConvNeXt-Tiny so carregados quando preciso
    r = diag.diagnosticar(img_bgr)                     # {"p", "circulos", "mascara", "p_doencas", "aviso", ...}
    cv2.imwrite("saida.jpg", desenhar(img_bgr, r))

Passo 1: o modelo de Hough (saudavel ou doenca). Passo 2: o modelo das doencas (hough_doencas.py): a probabilidade de
cada doenca e a media das arvores com a ConvNeXt-Tiny (models/bracol_convnext.pt); sem a rede (pesos ausentes ou
usar_convnext=False), responde so com as arvores, com os limiares delas, e avisa.
"""
from pathlib import Path

import cv2
import joblib
import numpy as np

import hough as rh
import hough_treinar as ht
from hough_dados import ARTEFATO_HOUGH

LADO_MAX = 1920  # fotos maiores sao reduzidas antes do SAM e do Hough
LADO_MIN = 32


def reduzir(img_bgr, lado_max=LADO_MAX):
    """Reduz a foto para o lado maior ter no maximo lado_max. Devolve (foto, fator = tamanho novo / original)."""
    maior = max(img_bgr.shape[:2])
    if maior <= lado_max:
        return img_bgr, 1.0
    f = lado_max / maior
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA), f


def _dispositivo():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Diagnosticador:
    """Modelo de Hough e modelo das doencas (+ ConvNeXt-Tiny, se adotada) prontos para fotos soltas.
    sam: funcao img_rgb -> (3 mascaras bool, 3 confiancas); convnext: funcao img_bgr -> (4,). Sem eles, o SAM
    (facebook/sam-vit-base) e a ConvNeXt-Tiny sao carregados na primeira foto que precisar, se usar_sam /
    usar_convnext.
    doencas: modelo das doencas (hough_doencas.py; dicionario ou caminho do .joblib). Sem ele, ou com o arquivo
    ausente, a resposta nao traz p_doencas."""

    def __init__(self, modelo=None, artefato=None, usar_sam=True, sam=None, disp=None, doencas=None,
                 usar_convnext=True, convnext=None, arquivo_convnext=None):
        artefato = artefato or ARTEFATO_HOUGH
        if modelo is None:
            if not artefato.exists():
                raise FileNotFoundError(f"{artefato} nao existe: rode hough_treinar.py cv e final")
            modelo = joblib.load(artefato)
        if doencas is not None and not isinstance(doencas, dict):
            doencas = joblib.load(doencas) if Path(doencas).exists() else None
        if doencas is not None and (doencas["param1"], doencas["param2"]) != (modelo["param1"], modelo["param2"]):
            raise ValueError("o modelo das doencas foi treinado com outro Hough (param1/param2) que o modelo de Hough")
        self.doencas = doencas
        self.modelo, self.sam, self.disp = modelo, sam, disp
        self.usar_sam = usar_sam or sam is not None
        self.convnext, self.usar_convnext = convnext, usar_convnext or convnext is not None
        self._convnext_tentada, self.arquivo_convnext = convnext is not None, arquivo_convnext

    def _contorno(self, img_bgr):
        """(mascara da folha, confianca, aviso). SAM com um ponto no centro; se ele nao achar a folha, a cor."""
        if self.usar_sam:
            if self.sam is None:
                from folha_sam import SegmentadorSAM
                self.disp = self.disp or _dispositivo()
                self.sam = SegmentadorSAM(self.disp)
            mascaras, confiancas = self.sam(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            m, conf = rh.escolher_mascara_hough(np.asarray(mascaras, bool), np.asarray(confiancas, float))
            if m is not None:
                return m, conf, ""
            motivo = "o SAM não achou uma folha no centro da foto"
        else:
            motivo = "SAM desligado"
        m = rh.mascara_vegetacao(img_bgr)
        if m.all():
            return m, 0.0, "nenhuma folha encontrada: a foto inteira foi analisada"
        return m, 0.0, f"contorno da folha estimado pela cor ({motivo})"

    def _probs_convnext(self, img_bgr):
        if not self.usar_convnext:
            return None
        if not self._convnext_tentada:
            self._convnext_tentada = True
            import convnext_bracol as cb
            self.disp = self.disp or _dispositivo()
            self.convnext = cb.previsor(self.arquivo_convnext or cb.ARTEFATO, self.disp)
        return None if self.convnext is None else np.asarray(self.convnext(img_bgr), float)

    def diagnosticar(self, img_bgr):
        """Diagnostico de uma foto BGR (uint8, qualquer tamanho e orientacao). Devolve:
        p ({classe: probabilidade} do modelo de Hough), limiar, positivo (bool: P(classe positiva do modelo, a
        doenca) >= limiar; o aplicativo decide o passo 1 pela maior das duas, em app.py),
        circulos [(x, y, raio, P(positiva) do circulo)] em px da foto recebida, mascara (bool, do tamanho da foto),
        confianca (do SAM; 0 = contorno pela cor), aviso (texto para o usuario, com acentos; vazio se nada a dizer),
        area_lesao (fracao da folha com mapa de lesao acima de 0,30: o atributo mapa_frac_030 da folha),
        circulos_classes ([{classe: probabilidade}] de cada circulo pelo nivel 1, na ordem de `circulos`) e, com o
        modelo das doencas, p_doencas ({doenca: probabilidade final}), limiares_doencas ({doenca: limiar}),
        p_doencas_arvores e p_doencas_convnext (as partes; a segunda None sem a rede) e circulos_doencas
        ([{classe: probabilidade}] de cada circulo para saudavel e cada doenca, pelo nivel 1 por doenca do modelo);
        sem o modelo das doencas, None."""
        if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3 \
                or min(img_bgr.shape[:2]) < LADO_MIN:
            raise ValueError(f"imagem inválida: precisa ser colorida e ter pelo menos {LADO_MIN} px de lado")
        img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
        trabalho, fator = reduzir(img_bgr)
        mascara, confianca, aviso = self._contorno(trabalho)
        avisos = [aviso] if aviso else []
        m = self.modelo
        e = rh.extrair(trabalho, mascara, confianca, m["param1"], m["param2"])
        dados = {"foto": {k: e[k] for k in ("circulos", "at_circulos", "at_folha")}}
        p = ht.prever(m, ["foto"], dados)[0]
        p_circ = np.zeros(len(e["circulos"]))
        n1 = m["nivel1"]
        classes = list(m.get("classes") or ht.CLASSES)  # as do conjunto em que o modelo foi treinado
        i_pos = classes.index(m.get("positiva", ht.POSITIVA))
        p_circ_classes = np.zeros((len(p_circ), len(classes)))  # P(classe) de cada circulo, na ordem de `classes`
        if len(p_circ):
            pp = n1.predict_proba(e["at_circulos"])
            for j, c in enumerate(n1.classes_):
                p_circ_classes[:, int(c)] = pp[:, j]
            p_circ = p_circ_classes[:, i_pos]
        if not len(p_circ):
            avisos.append("nenhum círculo candidato a lesão foi encontrado")
        limiar = m["limiar"]

        circ = rh.para_original(e["circulos"], e["folha"]) / fator
        h, w = img_bgr.shape[:2]
        if fator != 1.0:
            mascara = cv2.resize(mascara.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        nomes = lambda v: {c: float(v[j]) for j, c in enumerate(classes)}
        doencas = self._doencas(e, p_circ_classes, img_bgr)
        if doencas.pop("aviso", ""):
            avisos.append("ConvNeXt-Tiny indisponível: doenças só pelas árvores")
        return {"positivo": bool(p[i_pos] >= limiar), "p": nomes(p), "limiar": float(limiar),
                "circulos": [(float(x), float(y), float(r), float(pr)) for (x, y, r), pr in zip(circ, p_circ)],
                "mascara": mascara, "confianca": float(confianca), "aviso": "; ".join(avisos),
                "area_lesao": float(e["at_folha"][rh.NOMES_FOLHA.index("mapa_frac_030")]),
                "circulos_classes": [nomes(v) for v in p_circ_classes], **doencas}

    def _doencas(self, e, p_circ_classes, img_bgr):
        """p_doencas e limiares_doencas pelo modelo das doencas (as pontuacoes dos circulos sao as do nivel 1 do modelo
        de Hough, como no treino dele; em media com a ConvNeXt-Tiny) e circulos_doencas, a classe de cada circulo pelo
        nivel 1 por doenca (saudavel e cada doenca), usada so para colorir os circulos; sem o modelo, None. 'aviso' e
        True quando a rede nao esta disponivel (a resposta fica so com as arvores)."""
        if self.doencas is None:
            return {"p_doencas": None, "limiares_doencas": None, "p_doencas_arvores": None, "p_doencas_convnext": None,
                    "circulos_doencas": None}
        from hough_doencas import CLASSES5, combinar, pontuar5, prever_foto
        dados = {k: e[k] for k in ("circulos", "at_circulos", "at_folha")}
        p5 = pontuar5(self.doencas["nivel1_doencas"], ["foto"], {"foto": dados})["foto"]
        circulos = [{c: float(v) for c, v in zip(CLASSES5, linha)} for linha in p5]
        nomes = list(self.doencas["doencas"])
        p_arv = np.array([prever_foto(self.doencas, dados, p_circ_classes)[d] for d in nomes])
        p_cnn = self._probs_convnext(img_bgr)
        p, limiares, usou = combinar(self.doencas, p_arv, p_cnn)
        por_nome = lambda v: {d: float(x) for d, x in zip(nomes, v)}
        return {"p_doencas": por_nome(p), "limiares_doencas": por_nome(limiares), "p_doencas_arvores": por_nome(p_arv),
                "p_doencas_convnext": por_nome(p_cnn) if usou else None, "circulos_doencas": circulos,
                "aviso": not usou}


def desenhar(img_bgr, resultado):
    """Copia da foto com o contorno da folha (branco) e os circulos do Hough, do azul (nao parece da classe positiva)
    ao vermelho (parece)."""
    saida = img_bgr.copy()
    t = max(2, round(max(saida.shape[:2]) / 480))
    mascara = resultado["mascara"]
    if not mascara.all():
        contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(saida, contornos, -1, (255, 255, 255), t)
    for x, y, r, p in sorted(resultado["circulos"], key=lambda c: c[3]):
        cv2.circle(saida, (int(round(x)), int(round(y))), int(round(r)) + t, (int(255 * (1 - p)), 0, int(255 * p)), t)
    return saida
