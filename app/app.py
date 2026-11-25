"""Aplicativo local: envia fotos de folhas de cafe e mostra, para cada uma, se esta saudavel ou com doenca, quais
doencas aparecem e a severidade, com os circulos do Hough desenhados na foto.

Uso (na raiz do repositorio):  python -P app/app.py [--porta 5000] [--sem-sam] [--sem-convnext]
Depois abra http://127.0.0.1:5000 no navegador. So atende a propria maquina.

As fotos nao precisam de mascara: o contorno da folha e achado na hora (hough_inferir.py). Os modelos sao os do
BRACOL (models/bracol_hough.joblib, models/bracol_doencas.joblib e models/bracol_convnext.pt), carregados na primeira
foto enviada, com o SAM.

A resposta tem dois passos (ver classificar()): primeiro Saudavel ou Doenca; se for Doenca, aparecem a probabilidade
de cada doenca (modelo das doencas, em media com a ConvNeXt-Tiny) e a severidade estimada pela
area da folha com lesao (ver severidade()). Quando nem Saudavel nem Doenca passa de LIMITE_INCERTO, a resposta e
"incerto", com o palpite. Nenhuma doenca e privilegiada.
"""
import argparse
import base64
import io
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import Flask, render_template, request  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from comum import LIMITE_INCERTO, MODELOS  # noqa: E402

LADO_PAGINA = 1100  # lado maior da foto anotada mostrada na pagina
MAX_FOTOS = 12  # por envio
MAX_BYTES = 80 * 1024 * 1024  # por envio
NOMES = {"healthy": "Saudável", "disease": "Doença"}  # as classes do passo 1 (modelo de Hough do BRACOL)
TREINADO_EM = ("Modelos treinados em folhas de café arábica destacadas, fotografadas sobre fundo claro (BRACOL); em "
               "fotos tiradas na planta e em outras espécies o acerto é menor.")  # o rodape da pagina
NOMES_DOENCAS = {"miner": "Bicho-mineiro", "rust": "Ferrugem", "phoma": "Mancha de phoma",
                 "cercospora": "Cercosporiose"}  # as do modelo das doencas (hough_doencas.py)
ARTEFATO = MODELOS / "bracol_hough.joblib"  # passo 1
ARTEFATO_DOENCAS = MODELOS / "bracol_doencas.joblib"  # passo 2 (com a ConvNeXt-Tiny, models/bracol_convnext.pt)

# LIMITE_INCERTO (comum.py, 60%): "incerto" quando nem Saudavel nem Doenca passa disto. Os numeros fora da dobra e no
# teste estao em outputs/hough_bracol/resultados_teste.json.
SAUDAVEL = "healthy"
# cor de cada classe nos circulos desenhados na foto e na legenda (BGR). Com o modelo das doencas, cada doenca tem a
# sua cor (circulos_doencas de hough_inferir.py); sem ele, as classes do passo 1.
CORES_CIRCULO = {"healthy": (230, 140, 40),  # azul
                 "miner": (0, 200, 255),  # amarelo
                 "rust": (40, 40, 230),  # vermelho
                 "phoma": (215, 60, 225),  # magenta
                 "cercospora": (230, 225, 0),  # ciano
                 "disease": (0, 140, 255)}  # laranja (so sem o modelo das doencas)
CINZA = (160, 160, 160)  # circulo sem probabilidades por classe

# Severidade pela fracao da folha com lesao (area_lesao de hough_inferir.py), nas faixas de area com que o BRACOL foi
# rotulado (Esgario, Krohling e Ventura, 2020): saudavel < 0,1%, muito baixa 0,1-5%, baixa 5-10%, alta 10-15%,
# muito alta > 15%. Nenhum corte foi ajustado aos dados.
SEVERIDADES = ["Saudável", "Muito baixa", "Baixa", "Alta", "Muito alta"]
FAIXAS_SEVERIDADE = ["< 0,1%", "0,1 a 5%", "5 a 10%", "10 a 15%", "> 15%"]
CORTES_SEVERIDADE = (0.001, 0.05, 0.10, 0.15)
SEVERIDADE_SOBRE = ("Estimada pela parte da folha com cor de lesão, nas faixas de área em que o BRACOL foi rotulado. "
                    "Na validação do BRACOL (200 folhas), deu o nível rotulado em 83,5% das folhas e errou por no "
                    "máximo um nível em 98%; não foi medida no teste.")


def severidade(area):
    """Nivel de severidade (0 a 4) da fracao `area` da folha com lesao, nas faixas de CORTES_SEVERIDADE."""
    return int(np.searchsorted(CORTES_SEVERIDADE, area, side="right"))


def classificar(p):
    """{classe: probabilidade} -> {doenca (bool), p_saudavel, p_doenca, incerto}.
    Passo 1: Saudavel ou Doenca, pela maior das duas (P(Doenca) = 1 - P(saudavel)); incerto quando ela nao passa de
    LIMITE_INCERTO."""
    p_saudavel = float(p[SAUDAVEL])
    p_doenca = float(sum(v for c, v in p.items() if c != SAUDAVEL))
    return {"doenca": p_doenca > p_saudavel, "p_saudavel": p_saudavel, "p_doenca": p_doenca,
            "incerto": bool(max(p_saudavel, p_doenca) < LIMITE_INCERTO)}


def por_doenca(p_doencas, limiares):
    """Passo 2 com o modelo das doencas: [{chave, nome, p, limiar, provavel}] na ordem do modelo e os nomes das
    doencas provaveis (p >= limiar; uma folha pode ter mais de uma). Nenhuma passa do limiar: a de maior p."""
    linhas = [{"chave": d, "nome": NOMES_DOENCAS.get(d, d), "p": float(v), "limiar": float(limiares[d]),
               "provavel": bool(v >= limiares[d])} for d, v in p_doencas.items()]
    if not any(x["provavel"] for x in linhas):
        max(linhas, key=lambda x: x["p"])["provavel"] = True
    return linhas, [x["nome"] for x in linhas if x["provavel"]]


def cor_css(classe):
    """Cor de CORES_CIRCULO em #rrggbb, para a legenda da pagina."""
    b, g, r = CORES_CIRCULO.get(classe, CINZA)
    return f"#{r:02x}{g:02x}{b:02x}"


def classes_dos_circulos(r):
    """As classes com que os circulos sao coloridos, na ordem da legenda: saudavel e cada doenca do modelo das doencas
    (se ele pontua os circulos); senao, as classes do passo 1."""
    if r.get("circulos_doencas") is not None and r.get("p_doencas"):
        return [SAUDAVEL, *r["p_doencas"]]
    return list(r["p"])


def desenhar_classes(img_bgr, r):
    """Copia da foto com o contorno da folha (branco) e cada circulo do Hough na cor da classe com que ele mais se
    parece (CORES_CIRCULO): saudavel ou uma das doencas conhecidas, pelo nivel 1 por doenca (circulos_doencas de
    hough_inferir.py); sem ele, pelas classes do modelo de Hough (circulos_classes)."""
    saida = img_bgr.copy()
    t = max(2, round(max(saida.shape[:2]) / 480))
    mascara = r["mascara"]
    if not mascara.all():
        contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(saida, contornos, -1, (255, 255, 255), t)
    probs = r.get("circulos_doencas")
    if probs is None:
        probs = r.get("circulos_classes")
    probs = probs or [None] * len(r["circulos"])
    circulos = []
    for (x, y, raio, _), pc in zip(r["circulos"], probs):
        if pc:
            c = max(pc, key=pc.get)
            circulos.append((pc[c], x, y, raio, CORES_CIRCULO.get(c, CINZA)))
        else:
            circulos.append((0.0, x, y, raio, CINZA))
    for _, x, y, raio, cor in sorted(circulos, key=lambda c: c[0]):  # os mais seguros por cima
        cv2.circle(saida, (int(round(x)), int(round(y))), int(round(raio)) + t, cor, t)
    return saida


def ler_imagem(dados):
    """Bytes de um arquivo enviado -> foto BGR (uint8) em pe: aplica a rotacao do EXIF (fotos de celular) e leva
    transparencia e tons de cinza para 3 canais. ValueError se nao for uma imagem."""
    try:
        with Image.open(io.BytesIO(dados)) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except Exception:
        raise ValueError("não é uma imagem que eu consiga abrir (use JPG ou PNG)") from None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def para_data_uri(img_bgr, lado=LADO_PAGINA):
    """JPEG embutido na pagina, com o lado maior reduzido a `lado`."""
    h, w = img_bgr.shape[:2]
    f = min(1.0, lado / max(h, w))
    if f < 1.0:
        img_bgr = cv2.resize(img_bgr, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("não consegui gerar a imagem de saída")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def criar_app(diagnosticador=None, usar_sam=True, usar_convnext=True):
    """Aplicativo Flask. diagnosticador: objeto com .diagnosticar(img_bgr); sem ele, os modelos de verdade sao
    carregados na primeira foto."""
    app = Flask(__name__)
    app.context_processor(lambda: {"treinado_em": TREINADO_EM})  # o rodape de todas as respostas
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    estado = {"diag": diagnosticador}
    trava = threading.Lock()  # uma foto por vez: carregar e usar os modelos (SAM, ConvNeXt) nao e seguro em paralelo

    def diag():
        if estado["diag"] is None:
            from hough_inferir import Diagnosticador
            estado["diag"] = Diagnosticador(artefato=ARTEFATO, usar_sam=usar_sam, usar_convnext=usar_convnext,
                                            doencas=ARTEFATO_DOENCAS)
        return estado["diag"]

    def analisar(nome, dados):
        try:
            img = ler_imagem(dados)
            with trava:
                r = diag().diagnosticar(img)
        except ValueError as e:
            return {"nome": nome, "erro": str(e)}
        except Exception:  # falha do modelo (memoria da GPU, pesos ausentes...): o motivo vai para o terminal
            app.logger.exception("falha ao analisar %s", nome)
            return {"nome": nome, "interno": True,
                    "erro": "falha interna ao analisar esta foto (o motivo está no terminal do aplicativo)"}
        k = classificar(r["p"])
        sev = None
        if k["doenca"] and r.get("area_lesao") is not None:  # severidade so quando a resposta (ou o palpite) e Doenca
            nivel = severidade(r["area_lesao"])
            sev = {"nivel": nivel, "nome": SEVERIDADES[nivel], "area": r["area_lesao"],
                   "niveis": [{"nome": n, "faixa": f} for n, f in zip(SEVERIDADES, FAIXAS_SEVERIDADE)],
                   "sobre": SEVERIDADE_SOBRE}
        # passo 2, so mostrado se for Doenca: uma probabilidade por doenca (modelo das doencas)
        doencas, provaveis = (por_doenca(r["p_doencas"], r["limiares_doencas"]) if r.get("p_doencas")
                              else (None, []))
        return {"nome": nome, "erro": None, "doenca": k["doenca"], "incerto": k["incerto"], "limite": LIMITE_INCERTO,
                "resposta": "Doença" if k["doenca"] else NOMES[SAUDAVEL],
                "p_resposta": k["p_doenca"] if k["doenca"] else k["p_saudavel"],
                "p_saudavel": k["p_saudavel"], "p_doenca": k["p_doenca"],
                "doencas": doencas, "provaveis": provaveis, "com_convnext": bool(r.get("p_doencas_convnext")),
                "legenda": [{"chave": c, "nome": NOMES.get(c) or NOMES_DOENCAS.get(c, c), "cor": cor_css(c)}
                            for c in classes_dos_circulos(r)],
                "n_circulos": len(r["circulos"]), "aviso": r["aviso"],
                "imagem": para_data_uri(desenhar_classes(img, r)), "severidade": sev}

    def resumir(resultados):
        """Quantas fotos saudaveis, com doenca (e quantas com cada doenca; uma folha pode contar em mais de uma),
        incertas e que falharam."""
        validos = [r for r in resultados if not r["erro"]]
        certos = [r for r in validos if not r["incerto"]]
        doentes = [r for r in certos if r["doenca"]]
        nomes = []
        for r in validos:  # as doencas na ordem em que o modelo as mostra
            for n in [x["nome"] for x in r["doencas"] or []]:
                if n not in nomes:
                    nomes.append(n)
        return {"saudavel": len(certos) - len(doentes), "doenca": len(doentes),
                "tipos": [(n, sum(1 for r in doentes if n in r["provaveis"])) for n in nomes],
                "incerto": len(validos) - len(certos), "falhas": len(resultados) - len(validos)}

    @app.get("/")
    def inicio():
        return render_template("index.html", resultados=None, erro=None, aviso=None)

    @app.post("/")
    def enviar():
        arquivos = [f for f in request.files.getlist("fotos") if f and f.filename]
        if not arquivos:
            return render_template("index.html", resultados=None, erro="Escolha pelo menos uma foto.", aviso=None), 400
        aviso = None
        if len(arquivos) > MAX_FOTOS:
            aviso = f"Foram enviadas {len(arquivos)} fotos; analisei só as {MAX_FOTOS} primeiras."
        resultados = [analisar(f.filename, f.read()) for f in arquivos[:MAX_FOTOS]]
        if any(r.get("interno") for r in resultados):
            status = 500
        else:
            status = 400 if all(r["erro"] for r in resultados) else 200
        return render_template("index.html", resultados=resultados, resumo=resumir(resultados), erro=None,
                               aviso=aviso), status

    @app.errorhandler(413)
    def grande_demais(_):
        return render_template("index.html", resultados=None, aviso=None,
                               erro=f"Envio grande demais (limite de {MAX_BYTES // (1024 * 1024)} MB por vez)."), 413

    return app


def argumentos(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--porta", type=int, default=5000)
    ap.add_argument("--sem-sam", action="store_true", help="contorno da folha so pela cor (mais rapido, pior)")
    ap.add_argument("--sem-convnext", action="store_true",
                    help="quais doencas so pelas arvores, sem a ConvNeXt-Tiny (sem GPU, mais rapido)")
    return ap.parse_args(argv)


def main(argv=None):
    args = argumentos(argv)
    criar_app(usar_sam=not args.sem_sam, usar_convnext=not args.sem_convnext).run(host="127.0.0.1", port=args.porta)


if __name__ == "__main__":
    main()
