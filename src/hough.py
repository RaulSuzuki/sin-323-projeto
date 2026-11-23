"""Pipeline de Hough do RoCoLe: da foto de uma folha aos circulos candidatos a lesao e seus atributos.

O mesmo codigo serve ao treino (hough_treinar.py) e ao aplicativo (hough_inferir.py):

  foto + contorno da folha -> folha normalizada (raiz da area = LADO_FOLHA px, para que os raios valham em
  qualquer resolucao) -> mapa de lesao (quanto cada pixel da folha se afasta do verde rumo a amarelo, laranja ou
  marrom) -> transformada de Hough de circulos sobre o mapa -> atributos de cada circulo e da folha inteira.

Aqui o Hough e o detector de ONDE ha algo redondo e diferente do resto da folha. Dizer O QUE cada circulo e
(ferrugem, acaro, reflexo, nervura) fica para o classificador treinado com os atributos (hough_treinar.py).
So funcoes puras, sem torch.
"""
import sys
from dataclasses import dataclass

import cv2
import numpy as np

# Este arquivo se chamava rocole_hough.py; os modelos gravados antes da troca de nome (models/bracol_hough.joblib)
# guardam MediaDeModelos como rocole_hough.MediaDeModelos. O apelido deixa esses modelos carregarem.
sys.modules.setdefault("rocole_hough", sys.modules[__name__])

LADO_FOLHA = 700  # raiz da area da folha depois de normalizar, em px (mediana do RoCoLe em 1920x1080: ~705)
MARGEM = 0.10  # margem da caixa da folha, como em folha_sam.py
ESCALA_MIN, ESCALA_MAX = 0.2, 2.0
RAIO_MIN, RAIO_MAX = 4, 48  # raio das lesoes, em px normalizados
MAX_CIRCULOS = 60  # ficam os de mais votos
PARAM1, PARAM2 = 60, 12  # Canny alto e limiar de votos do Hough (baixo de proposito: o classificador filtra depois)
DIST_MIN = 10  # distancia minima entre centros
BORDA = 3  # px da borda da folha ignorados (o contorno nunca e exato)
GRAUS_CHEIO = 60.0  # desvio de matiz que satura o mapa (verde -> laranja/marrom)
CROMA_MIN, CROMA_FAIXA = 6.0, 10.0  # pixels quase sem cor (reflexo, sombra) nao contam
PERCENTIL_REF = 75  # matiz de referencia = o quarto mais verde da folha (resiste a folha muito amarelada)


@dataclass
class FolhaNorm:
    img: np.ndarray  # BGR uint8, recorte normalizado
    mascara: np.ndarray  # bool, mesma forma
    escala: float  # px normalizado / px original
    caixa: tuple  # (x0, y0, x1, y1) do recorte na foto original


def mascara_vegetacao(img_bgr):
    """Contorno da folha so pela cor, para quando nao ha mascara do SAM: a regiao verde/amarela ligada ao centro
    (ou a maior), com os buracos preenchidos (lesoes laranja/marrom dentro da folha continuam nela).
    Sem vegetacao bastante, devolve a foto inteira."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    veg = ((hsv[..., 0] >= 20) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 40)).astype(np.uint8)
    k = max(3, int(round(0.012 * max(h, w))) | 1)
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(veg, connectivity=8)
    if n <= 1:
        return np.ones((h, w), bool)
    r = int(rotulos[h // 2, w // 2]) or 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    if stats[r, cv2.CC_STAT_AREA] < 0.02 * h * w:
        return np.ones((h, w), bool)
    contornos, _ = cv2.findContours((rotulos == r).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cheia = np.zeros((h, w), np.uint8)
    cv2.drawContours(cheia, contornos, -1, 1, thickness=cv2.FILLED)
    return cheia.astype(bool)


class MediaDeModelos:
    """Media das probabilidades de varios classificadores ajustados nos mesmos dados (nivel 2 'media' de
    hough_treinar.py). Fica neste modulo, que nunca roda como script, para que o modelo gravado com ela
    possa ser carregado por qualquer programa (teste, aplicativo)."""

    def __init__(self, modelos):
        self.modelos = modelos

    def fit(self, x, y):
        for m in self.modelos:
            m.fit(x, y)
        self.classes_ = self.modelos[0].classes_
        return self

    def predict_proba(self, x):
        return np.mean([m.predict_proba(x) for m in self.modelos], axis=0)


def escolher_mascara_hough(mascaras, confiancas):
    """Das 3 mascaras do SAM (ponto no centro), fica o pedaco ligado ao centro da de maior confianca com area valida
    (regra de folha_sam.escolher_mascara), MESMO com confianca baixa: para o Hough, um contorno imperfeito vale
    mais que a foto inteira, com solo e folhas secas. Devolve (mascara ou None, confianca); a confianca vira
    atributo da folha."""
    from folha_sam import componente_do_centro, escolher_mascara
    _, info = escolher_mascara(mascaras, confiancas)
    if info["indice"] < 0:
        return None, 0.0
    h, w = mascaras.shape[1:]
    return componente_do_centro(mascaras[info["indice"]], (w // 2, h // 2)), float(info["confianca"])


def normalizar_folha(img_bgr, mascara):
    """Recorta a caixa da folha (+ MARGEM) e redimensiona para que a raiz da area da folha seja LADO_FOLHA px.
    Mascara vazia: a foto inteira e a folha."""
    h, w = img_bgr.shape[:2]
    mascara = np.asarray(mascara, bool)
    if not mascara.any():
        mascara = np.ones((h, w), bool)
    escala = float(np.clip(LADO_FOLHA / np.sqrt(float(mascara.sum())), ESCALA_MIN, ESCALA_MAX))
    x, y, bw, bh = cv2.boundingRect(mascara.astype(np.uint8))
    dx, dy = round(MARGEM * bw), round(MARGEM * bh)
    x0, y0, x1, y1 = max(0, x - dx), max(0, y - dy), min(w, x + bw + dx), min(h, y + bh + dy)
    nl, na = max(1, round((x1 - x0) * escala)), max(1, round((y1 - y0) * escala))
    img_n = cv2.resize(img_bgr[y0:y1, x0:x1], (nl, na), interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC)
    m_n = cv2.resize(mascara[y0:y1, x0:x1].astype(np.uint8), (nl, na), interpolation=cv2.INTER_NEAREST) > 0
    return FolhaNorm(np.ascontiguousarray(img_n), m_n, escala, (x0, y0, x1, y1))


def _lab(img_bgr):
    """L*, a*, b* (float32, a* e b* centrados em 0), angulo de matiz em graus e croma."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1:] -= 128.0
    return lab, np.degrees(np.arctan2(lab[..., 2], lab[..., 1])), np.hypot(lab[..., 1], lab[..., 2])


def _miolo(mascara):
    """A mascara sem os BORDA px da beirada."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BORDA + 1, 2 * BORDA + 1))
    return cv2.erode(mascara.astype(np.uint8), k) > 0


def mapa_lesao(folha):
    """Mapa 0-255 de 'quanto este pixel parece lesao': desvio do angulo de matiz (L*a*b*) em relacao ao verde da
    propria folha, rumo a amarelo/laranja/marrom, pesado pelo croma (reflexos e sombras, quase sem cor, ficam
    perto de 0). Suavizado (sigma 1,5) para o Canny do Hough. Zero fora da folha e na beirada."""
    _, matiz, croma = _lab(folha.img)
    dentro = _miolo(folha.mascara)
    validos = dentro & (croma > CROMA_MIN)
    if int(validos.sum()) < 50:
        return np.zeros(folha.mascara.shape, np.uint8)
    ref = np.percentile(matiz[validos], PERCENTIL_REF)
    desvio = (ref - matiz + 180.0) % 360.0 - 180.0  # positivo = mais para o vermelho do que a folha
    v = np.clip(desvio / GRAUS_CHEIO, 0, 1) * np.clip((croma - CROMA_MIN) / CROMA_FAIXA, 0, 1)
    v[~dentro] = 0
    v = cv2.GaussianBlur(v.astype(np.float32), (0, 0), 1.5)
    v[~dentro] = 0
    return np.rint(v * 255).astype(np.uint8)


def propor_circulos(folha, mapa, param1=PARAM1, param2=PARAM2):
    """Transformada de Hough de circulos (metodo do gradiente) sobre o mapa de lesao. Devolve (N, 3) float32
    (x, y, raio) em px normalizados, na ordem do OpenCV (mais votos primeiro), so com centro dentro da folha,
    no maximo MAX_CIRCULOS."""
    c = cv2.HoughCircles(mapa, cv2.HOUGH_GRADIENT, dp=1, minDist=DIST_MIN, param1=param1, param2=param2,
                         minRadius=RAIO_MIN, maxRadius=RAIO_MAX)
    if c is None:
        return np.zeros((0, 3), np.float32)
    c = c[0, :, :3].astype(np.float32)
    h, w = folha.mascara.shape
    xi, yi = np.rint(c[:, 0]).astype(int), np.rint(c[:, 1]).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    ok[ok] = folha.mascara[yi[ok], xi[ok]]
    return c[ok][:MAX_CIRCULOS]


def para_original(circ, folha):
    """Circulos (x, y, raio) em px normalizados -> px da foto original."""
    o = np.asarray(circ, np.float32).reshape(-1, 3).copy()
    o[:, 0] = o[:, 0] / folha.escala + folha.caixa[0]
    o[:, 1] = o[:, 1] / folha.escala + folha.caixa[1]
    o[:, 2] = o[:, 2] / folha.escala
    return o


# ---------------------------------------------------------------- atributos (extracao de caracteristicas)
N_ANGULOS = 36  # pontos da circunferencia em que se procura borda (apoio do Hough)
NIVEIS_GLCM = 16
LIMIAR_BLOB_MIN = 40  # no mapa 0-255

NOMES_CIRCULO = (
    ["raio", "dist_borda", "apoio", "ordem"]  # geometria e apoio do Hough
    + ["L", "a", "b", "croma", "dp_L", "dp_a", "dp_b", "mapa_media", "mapa_frac", "mapa_max"]  # cor no disco
    + ["delta_L", "delta_a", "delta_b", "delta_mapa"]  # disco menos o anel em volta (r a 2r)
    + ["centro_L", "centro_a", "centro_b"]  # miolo (r/2) menos a coroa do disco: centro marrom com halo amarelo
    + ["variancia"] + [f"lbp_{i}" for i in range(10)]  # textura
    + ["glcm_contraste", "glcm_homogeneidade", "glcm_energia", "glcm_correlacao"]
    + ["blob_area", "blob_circularidade", "blob_alongamento", "blob_solidez"])  # forma da mancha sob o circulo
NOMES_FOLHA = (
    ["folha_L", "folha_a", "folha_b", "folha_dp_L", "folha_dp_a", "folha_dp_b", "folha_croma"]
    + ["matiz_ref", "matiz_dp", "mapa_media_folha", "mapa_frac_015", "mapa_frac_030", "mapa_frac_050", "mapa_frac_070"]
    + ["folha_variancia"] + [f"folha_lbp_{i}" for i in range(10)]
    + ["confianca", "fracao_area"])

_VIZINHOS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]  # em volta, no sentido horario


def _tabela_lbp():
    """Codigo de 8 bits -> bin do LBP uniforme: numero de uns (0-8) se ha no maximo 2 transicoes; senao, bin 9."""
    t = np.empty(256, np.uint8)
    for c in range(256):
        bits = [(c >> i) & 1 for i in range(8)]
        trans = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        t[c] = sum(bits) if trans <= 2 else 9
    return t


_LBP = _tabela_lbp()


def codigos_lbp(cinza):
    """Bin do LBP uniforme (8 vizinhos, raio 1) de cada pixel: 0-9, uint8."""
    g = cv2.copyMakeBorder(cinza, 1, 1, 1, 1, cv2.BORDER_REPLICATE).astype(np.int16)
    h, w = cinza.shape
    centro = g[1:-1, 1:-1]
    cod = np.zeros((h, w), np.uint8)
    for i, (dy, dx) in enumerate(_VIZINHOS):
        cod |= (g[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] >= centro).astype(np.uint8) << i
    return _LBP[cod]


def histograma_lbp(codigos, onde):
    """Histograma normalizado (10 bins) dos codigos dentro de `onde`; zeros se a regiao e vazia."""
    h = np.bincount(codigos[onde], minlength=10).astype(np.float32)
    return h / h.sum() if h.sum() else h


def glcm(cinza):
    """Contraste, homogeneidade, energia e correlacao da matriz de coocorrencia (16 niveis, vizinho a direita e
    abaixo, simetrica). Area lisa: contraste 0, homogeneidade 1, energia 1, correlacao 1."""
    q = (cinza.astype(np.int32) * NIVEIS_GLCM) // 256
    n = NIVEIS_GLCM
    pares = np.concatenate([q[:, :-1].ravel() * n + q[:, 1:].ravel(), q[:-1, :].ravel() * n + q[1:, :].ravel()])
    if pares.size == 0:
        return np.array([0, 1, 1, 1], np.float32)
    p = np.bincount(pares, minlength=n * n).reshape(n, n).astype(np.float64)
    p = (p + p.T) / (2 * p.sum())
    i, j = np.indices((n, n))
    mu = float((p * i).sum())
    var = float((p * (i - mu) ** 2).sum())
    cor = float((p * (i - mu) * (j - mu)).sum() / var) if var > 1e-9 else 1.0
    escala = (255.0 / (n - 1)) ** 2  # contraste na escala 0-255
    return np.array([(p * (i - j) ** 2).sum() * escala, (p / (1 + (i - j) ** 2)).sum(), (p ** 2).sum(), cor], np.float32)


def _camadas(folha, mapa):
    """Tudo o que se calcula uma vez por folha e os atributos reaproveitam."""
    lab, matiz, croma = _lab(folha.img)
    cinza = cv2.cvtColor(folha.img, cv2.COLOR_BGR2GRAY)
    f = cinza.astype(np.float32)
    variancia = np.maximum(cv2.blur(f * f, (5, 5)) - cv2.blur(f, (5, 5)) ** 2, 0)
    bordas = cv2.dilate(cv2.Canny(mapa, PARAM1 // 2, PARAM1), np.ones((5, 5), np.uint8)) > 0  # +-2 px de folga
    dist = cv2.distanceTransform(folha.mascara.astype(np.uint8), cv2.DIST_L2, 3)
    return {"lab": lab, "matiz": matiz, "croma": croma, "cinza": cinza, "variancia": variancia, "bordas": bordas,
            "lbp": codigos_lbp(cinza), "dist": dist, "mapa": mapa.astype(np.float32) / 255.0}


def _media(v, onde, padrao=0.0):
    return float(v[onde].mean()) if onde.any() else padrao


def _forma_blob(mapa_janela, disco, raio):
    """Area (em areas do circulo), circularidade 4*pi*A/P^2, alongamento (eixo menor / maior) e solidez (area /
    fecho convexo) da mancha do mapa que mais se sobrepoe ao disco. Sem mancha: zeros."""
    limiar = max(LIMIAR_BLOB_MIN, 0.5 * 255.0 * _media(mapa_janela, disco))
    binario = (mapa_janela * 255.0 >= limiar).astype(np.uint8)
    n, rotulos = cv2.connectedComponents(binario, connectivity=8)
    if n <= 1:
        return [0.0, 0.0, 0.0, 0.0]
    sobre = np.bincount(rotulos[disco], minlength=n)
    sobre[0] = 0
    if sobre.max() == 0:
        return [0.0, 0.0, 0.0, 0.0]
    comp = (rotulos == int(sobre.argmax())).astype(np.uint8)
    contornos, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contornos, key=cv2.contourArea)
    area = float(comp.sum())
    per = cv2.arcLength(c, True)
    m = cv2.moments(comp, binaryImage=True)
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    raiz = np.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11 ** 2, 0.0))
    l1, l2 = (mu20 + mu02 + raiz) / 2, (mu20 + mu02 - raiz) / 2
    fecho = cv2.contourArea(cv2.convexHull(c))
    return [min(area / (np.pi * raio ** 2), 4.0), float(min(4 * np.pi * area / per ** 2, 1.0)) if per > 0 else 0.0,
            float(np.sqrt(max(l2, 0.0) / l1)) if l1 > 1e-9 else 0.0, float(min(area / fecho, 1.0)) if fecho > 0 else 1.0]


def atributos_circulos(folha, mapa, circ, camadas=None):
    """Atributos de cada circulo, na ordem de NOMES_CIRCULO: (N, len(NOMES_CIRCULO)) float32."""
    circ = np.asarray(circ, np.float32).reshape(-1, 3)
    saida = np.zeros((len(circ), len(NOMES_CIRCULO)), np.float32)
    if not len(circ):
        return saida
    k = camadas or _camadas(folha, mapa)
    h, w = folha.mascara.shape
    ang = np.linspace(0, 2 * np.pi, N_ANGULOS, endpoint=False)
    for n, (x, y, r) in enumerate(circ):
        r = max(float(r), 1.0)
        x0, x1 = int(max(0, np.floor(x - 2 * r - 2))), int(min(w, np.ceil(x + 2 * r + 3)))
        y0, y1 = int(max(0, np.floor(y - 2 * r - 2))), int(min(h, np.ceil(y + 2 * r + 3)))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - x, yy - y)
        folha_j = folha.mascara[y0:y1, x0:x1]
        disco, anel = (d <= r) & folha_j, (d > r) & (d <= 2 * r) & folha_j
        miolo, coroa = (d <= r / 2) & folha_j, (d > r / 2) & (d <= r) & folha_j
        if not disco.any():
            continue
        lab, mapa_j = k["lab"][y0:y1, x0:x1], k["mapa"][y0:y1, x0:x1]
        px = np.clip(np.rint(x + r * np.cos(ang)).astype(int), 0, w - 1)
        py = np.clip(np.rint(y + r * np.sin(ang)).astype(int), 0, h - 1)
        xc, yc = int(np.clip(round(float(x)), 0, w - 1)), int(np.clip(round(float(y)), 0, h - 1))
        geometria = [r, float(min(k["dist"][yc, xc], 100.0)), float(k["bordas"][py, px].mean()), n / MAX_CIRCULOS]
        cor = [float(lab[..., i][disco].mean()) for i in range(3)] + [_media(k["croma"][y0:y1, x0:x1], disco)]
        cor += [float(lab[..., i][disco].std()) for i in range(3)]
        cor += [float(mapa_j[disco].mean()), float((mapa_j[disco] > 0.3).mean()), float(mapa_j[disco].max())]
        ref = [_media(lab[..., i], anel, cor[i]) for i in range(3)]
        contraste = [cor[i] - ref[i] for i in range(3)] + [cor[7] - _media(mapa_j, anel, cor[7])]
        radial = [_media(lab[..., i], miolo, cor[i]) - _media(lab[..., i], coroa, cor[i]) for i in range(3)]
        bx0, bx1 = int(max(0, np.floor(x - r))), int(min(w, np.ceil(x + r + 1)))
        by0, by1 = int(max(0, np.floor(y - r))), int(min(h, np.ceil(y + r + 1)))
        textura = [_media(k["variancia"][y0:y1, x0:x1], disco)]
        textura += histograma_lbp(k["lbp"][y0:y1, x0:x1], disco).tolist() + glcm(k["cinza"][by0:by1, bx0:bx1]).tolist()
        saida[n] = geometria + cor + contraste + radial + textura + _forma_blob(mapa_j, disco, r)
    return np.nan_to_num(saida, nan=0.0, posinf=0.0, neginf=0.0)


def atributos_folha(folha, mapa, confianca, fracao_area, camadas=None):
    """Atributos da folha inteira, na ordem de NOMES_FOLHA. confianca: a do SAM (0 = contorno so pela cor);
    fracao_area: parte da foto original que a folha ocupa."""
    k = camadas or _camadas(folha, mapa)
    dentro = _miolo(folha.mascara)
    if not dentro.any():
        dentro = folha.mascara
    lab, m = k["lab"], k["mapa"][dentro]
    cor = [float(lab[..., i][dentro].mean()) for i in range(3)] + [float(lab[..., i][dentro].std()) for i in range(3)]
    validos = dentro & (k["croma"] > CROMA_MIN)
    matiz = k["matiz"][validos] if validos.any() else np.zeros(1, np.float32)
    saida = cor + [float(k["croma"][dentro].mean()), float(np.percentile(matiz, PERCENTIL_REF)), float(matiz.std()),
                   float(m.mean())] + [float((m > t).mean()) for t in (0.15, 0.30, 0.50, 0.70)]
    saida += [float(k["variancia"][dentro].mean())] + histograma_lbp(k["lbp"], dentro).tolist()
    saida += [float(confianca), float(fracao_area)]
    return np.nan_to_num(np.asarray(saida, np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extrair(img_bgr, mascara=None, confianca=0.0, param1=PARAM1, param2=PARAM2):
    """Da foto aos circulos e atributos. mascara: contorno da folha (bool, do tamanho da foto) e sua confianca;
    sem mascara, o contorno sai so da cor (mascara_vegetacao) e a confianca e 0."""
    if mascara is None:
        mascara, confianca = mascara_vegetacao(img_bgr), 0.0
    mascara = np.asarray(mascara, bool)
    if not mascara.any():
        mascara = np.ones(img_bgr.shape[:2], bool)
    folha = normalizar_folha(img_bgr, mascara)
    mapa = mapa_lesao(folha)
    circ = propor_circulos(folha, mapa, param1, param2)
    k = _camadas(folha, mapa)
    return {"folha": folha, "mapa": mapa, "circulos": circ, "confianca": float(confianca),
            "at_circulos": atributos_circulos(folha, mapa, circ, k),
            "at_folha": atributos_folha(folha, mapa, confianca, float(mascara.mean()), k)}
