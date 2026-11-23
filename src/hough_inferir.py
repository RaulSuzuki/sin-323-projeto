"""Aplica o modelo de Hough do RoCoLe a uma foto qualquer de folha, SEM mascara: o contorno da folha e calculado
aqui (SAM com um ponto no centro; se o SAM nao achar a folha, pela cor). Usado pelo aplicativo (app/app.py) e pela
conferencia externa (hough_avaliar.py --conferencia-externa).

    diag = Diagnosticador()                 # carrega models/rocole_hough.joblib; SAM e ResNet-50 so quando preciso
    r = diag.diagnosticar(img_bgr)          # {"ferrugem", "p", "limiar", "circulos", "mascara", "aviso", ...}
    cv2.imwrite("saida.jpg", desenhar(img_bgr, r))

Se o modelo gravado tem fusao (hough_treinar.py fusao), a resposta e a media ponderada das probabilidades do
modelo de Hough e da ResNet-50 (models/rocole_resnet50.pt), com o limiar da fusao; sem os pesos da ResNet-50, responde
so o modelo de Hough, com o limiar dele, e avisa.
"""
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image

import hough as rh
import hough_treinar as ht
from comum import ROCOLE_ALTURA, ROCOLE_LARGURA

LADO_MAX = 1920  # fotos maiores sao reduzidas antes do SAM e do Hough (o RoCoLe e 1920x1080)
LADO_MIN = 32


def reduzir(img_bgr, lado_max=LADO_MAX):
    """Reduz a foto para o lado maior ter no maximo lado_max. Devolve (foto, fator = tamanho novo / original)."""
    maior = max(img_bgr.shape[:2])
    if maior <= lado_max:
        return img_bgr, 1.0
    f = lado_max / maior
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA), f


def entrada_resnet(img_bgr):
    """Foto no formato que a ResNet-50 viu no treino: deitada, 16:9 (corta o centro), 512x288, RGB (PIL)."""
    if img_bgr.shape[0] > img_bgr.shape[1]:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    h, w = img_bgr.shape[:2]
    alvo = ROCOLE_LARGURA / ROCOLE_ALTURA
    if w / h > alvo:
        nw = max(1, round(h * alvo))
        img_bgr = img_bgr[:, (w - nw) // 2:(w - nw) // 2 + nw]
    else:
        nh = max(1, round(w / alvo))
        img_bgr = img_bgr[(h - nh) // 2:(h - nh) // 2 + nh]
    rgb = Image.fromarray(cv2.cvtColor(np.ascontiguousarray(img_bgr), cv2.COLOR_BGR2RGB))
    return rgb.resize((ROCOLE_LARGURA, ROCOLE_ALTURA), Image.LANCZOS)


def _dispositivo():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _carregar_resnet(disp):
    """Funcao img_bgr -> probabilidades (3,) da ResNet-50 final; None se os pesos nao estao em models/."""
    import torch

    from cnn_modelo import carregar, transformacao
    try:
        modelo = carregar("resnet50", disp)
    except FileNotFoundError:
        return None
    tf = transformacao(False)

    def prever(img_bgr):
        x = tf(entrada_resnet(img_bgr)).unsqueeze(0).to(disp)
        with torch.no_grad(), torch.autocast(device_type=disp.type, enabled=disp.type == "cuda"):
            saida = modelo(x)
        return torch.softmax(saida.float(), 1)[0].cpu().numpy()

    return prever


class Diagnosticador:
    """Modelo de Hough (+ fusao com a ResNet-50, se adotada) pronto para fotos soltas.
    sam: funcao img_rgb -> (3 mascaras bool, 3 confiancas); resnet: funcao img_bgr -> (3,). Sem eles, o SAM
    (facebook/sam-vit-base) e a ResNet-50 sao carregados na primeira foto, se usar_sam / usar_resnet.
    doencas: modelo das doencas (hough_doencas.py; dicionario ou caminho do .joblib). Sem ele, ou com o arquivo
    ausente, a resposta nao traz p_doencas."""

    def __init__(self, modelo=None, artefato=None, usar_sam=True, usar_resnet=True, sam=None, resnet=None, disp=None,
                 doencas=None):
        artefato = artefato or ht.ARTEFATO
        if modelo is None:
            if not artefato.exists():
                raise FileNotFoundError(f"{artefato} nao existe: rode hough_treinar.py cv e final")
            modelo = joblib.load(artefato)
        if doencas is not None and not isinstance(doencas, dict):
            doencas = joblib.load(doencas) if Path(doencas).exists() else None
        if doencas is not None and (doencas["param1"], doencas["param2"]) != (modelo["param1"], modelo["param2"]):
            raise ValueError("o modelo das doencas foi treinado com outro Hough (param1/param2) que o modelo de Hough")
        self.doencas = doencas
        self.modelo, self.sam, self.resnet, self.disp = modelo, sam, resnet, disp
        self.usar_sam, self.usar_resnet = usar_sam or sam is not None, usar_resnet or resnet is not None
        self._resnet_tentada = resnet is not None

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

    def _probs_resnet(self, img_bgr):
        if not self.usar_resnet:
            return None
        if not self._resnet_tentada:
            self._resnet_tentada = True
            self.disp = self.disp or _dispositivo()
            self.resnet = _carregar_resnet(self.disp)
        return None if self.resnet is None else np.asarray(self.resnet(img_bgr), float)

    def diagnosticar(self, img_bgr):
        """Diagnostico de uma foto BGR (uint8, qualquer tamanho e orientacao). Devolve:
        ferrugem (bool: de que lado do limiar a foto caiu), incerto (bool: P(ferrugem) dentro da faixa de recusa,
        limiar +- margem; a resposta e "mostre a um agronomo"), margem (0 se o modelo nao tem recusa),
        p / p_hough / p_resnet ({classe: probabilidade}; p_resnet None sem fusao), limiar,
        circulos [(x, y, raio, P(ferrugem) do circulo)] em px da foto recebida, mascara (bool, do tamanho da foto),
        confianca (do SAM; 0 = contorno pela cor), aviso (texto para o usuario, com acentos; vazio se nada a dizer),
        area_lesao (fracao da folha com mapa de lesao acima de 0,30: o atributo mapa_frac_030 da folha),
        circulos_classes ([{classe: probabilidade}] de cada circulo pelo nivel 1, na ordem de `circulos`) e, com o
        modelo das doencas, p_doencas ({doenca: probabilidade}), limiares_doencas ({doenca: limiar}) e
        circulos_doencas ([{classe: probabilidade}] de cada circulo para saudavel e cada doenca, pelo nivel 1 por doenca
        do modelo; None se o modelo nao o tem); sem o modelo, None."""
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
        p_hough = ht.prever(m, ["foto"], dados)[0]
        p_circ = np.zeros(len(e["circulos"]))
        n1 = m.get("nivel1")
        classes = m.get("classes") or ht.CLASSES  # as do conjunto em que o modelo foi treinado
        p_circ_classes = np.zeros((len(p_circ), len(classes)))  # P(classe) de cada circulo, na ordem de `classes`
        if n1 is not None and len(p_circ):
            pp = n1.predict_proba(e["at_circulos"])
            for j, c in enumerate(n1.classes_):
                p_circ_classes[:, int(c)] = pp[:, j]
        if n1 is not None and len(p_circ) and ht.RUST in list(n1.classes_):
            p_circ = n1.predict_proba(e["at_circulos"])[:, list(n1.classes_).index(ht.RUST)]
        if not len(p_circ):
            avisos.append("nenhum círculo candidato a lesão foi encontrado")

        fusao, p_resnet = m.get("fusao"), None
        if fusao:
            p_resnet = self._probs_resnet(img_bgr)
            if p_resnet is None:
                avisos.append("ResNet-50 indisponível: resposta só do modelo de Hough")
        if p_resnet is not None:
            p, limiar = ht.fundir(p_hough, p_resnet, fusao["w"]), fusao["limiar"]
        else:
            p, limiar = p_hough, m["limiar"]

        # recusa (hough_recusa.py): a faixa foi escolhida para as probabilidades da fusao; sem a ResNet-50 nao vale
        recusa = m.get("recusa") if p_resnet is not None else None
        margem = float(recusa["margem"]) if recusa else 0.0
        incerto = bool(abs(p[ht.RUST] - limiar) < margem)

        circ = rh.para_original(e["circulos"], e["folha"]) / fator
        h, w = img_bgr.shape[:2]
        if fator != 1.0:
            mascara = cv2.resize(mascara.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        nomes = lambda v: {c: float(v[j]) for j, c in enumerate(classes)}
        return {"ferrugem": bool(p[ht.RUST] >= limiar), "incerto": incerto, "margem": margem,
                "p": nomes(p), "p_hough": nomes(p_hough),
                "p_resnet": None if p_resnet is None else nomes(p_resnet), "limiar": float(limiar),
                "circulos": [(float(x), float(y), float(r), float(pr)) for (x, y, r), pr in zip(circ, p_circ)],
                "mascara": mascara, "confianca": float(confianca), "aviso": "; ".join(avisos),
                "area_lesao": float(e["at_folha"][rh.NOMES_FOLHA.index("mapa_frac_030")]),
                "circulos_classes": [nomes(v) for v in p_circ_classes],
                **self._doencas(e, p_circ_classes)}

    def _doencas(self, e, p_circ_classes):
        """p_doencas e limiares_doencas pelo modelo das doencas (as pontuacoes dos circulos sao as do nivel 1 do modelo
        de Hough, como no treino dele) e circulos_doencas, a classe de cada circulo pelo nivel 1 por doenca (saudavel e
        cada doenca), usada so para colorir os circulos; sem o modelo, None."""
        if self.doencas is None:
            return {"p_doencas": None, "limiares_doencas": None, "circulos_doencas": None}
        from hough_doencas import CLASSES5, pontuar5, prever_foto
        dados = {k: e[k] for k in ("circulos", "at_circulos", "at_folha")}
        n1 = self.doencas.get("nivel1_doencas")
        circulos = None
        if n1 is not None:
            p5 = pontuar5(n1, ["foto"], {"foto": dados})["foto"]
            circulos = [{c: float(v) for c, v in zip(CLASSES5, linha)} for linha in p5]
        return {"p_doencas": prever_foto(self.doencas, dados, p_circ_classes),
                "limiares_doencas": dict(zip(self.doencas["doencas"], map(float, self.doencas["limiares"]))),
                "circulos_doencas": circulos}


def desenhar(img_bgr, resultado):
    """Copia da foto com o contorno da folha (branco) e os circulos do Hough, do azul (nao parece ferrugem) ao
    vermelho (parece ferrugem)."""
    saida = img_bgr.copy()
    t = max(2, round(max(saida.shape[:2]) / 480))
    mascara = resultado["mascara"]
    if not mascara.all():
        contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(saida, contornos, -1, (255, 255, 255), t)
    for x, y, r, p in sorted(resultado["circulos"], key=lambda c: c[3]):
        cv2.circle(saida, (int(round(x)), int(round(y))), int(round(r)) + t, (int(255 * (1 - p)), 0, int(255 * p)), t)
    return saida
