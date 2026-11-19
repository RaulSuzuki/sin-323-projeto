"""Aplicativo local: envia fotos de folhas de cafe e mostra se o modelo ve ferrugem, com os circulos do Hough.

Uso (na raiz do repositorio):  python -P app/app.py [--porta 5000] [--modelo bracol|rocole] [--sem-sam] [--sem-resnet]
Depois abra http://127.0.0.1:5000 no navegador. So atende a propria maquina.

As fotos nao precisam de mascara: o contorno da folha e achado na hora (rocole_hough_inferir.py). O modelo e o SAM
sao carregados na primeira foto enviada.

--modelo escolhe em que conjunto o modelo de Hough foi treinado:
  bracol (padrao): models/bracol_hough.joblib, so o modelo de Hough; o que menos da alarme em folha sem ferrugem.
  rocole: models/rocole_hough.joblib, com a fusao com a ResNet-50 e a resposta "incerto" (fotos de robusta na planta).
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

LADO_PAGINA = 1100  # lado maior da foto anotada mostrada na pagina
MAX_FOTOS = 12  # por envio
MAX_BYTES = 80 * 1024 * 1024  # por envio
NOMES = {"healthy": "Saudável", "red_spider_mite": "Ácaro-vermelho", "other": "Outra lesão", "rust": "Ferrugem"}
MODELO_PADRAO = "bracol"
TREINADO_EM = {  # o que o rodape da pagina diz sobre o modelo carregado
    "bracol": "Modelo treinado em folhas de café arábica destacadas, fotografadas sobre fundo claro (BRACOL); em "
              "fotos tiradas na planta e em outras espécies o acerto é menor.",
    "rocole": "Modelo treinado em fotos de café robusta tiradas na planta (RoCoLe); em outras espécies e tipos de "
              "foto o acerto é menor.",
}


def caminho_modelo(nome):
    """Arquivo do modelo de Hough treinado no conjunto `nome` (bracol ou rocole)."""
    from rocole_hough_dados import carregar_conjunto
    return carregar_conjunto(nome).artefato


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


def criar_app(diagnosticador=None, usar_sam=True, usar_resnet=True, modelo=MODELO_PADRAO):
    """Aplicativo Flask. diagnosticador: objeto com .diagnosticar(img_bgr); sem ele, o modelo de verdade (o do
    conjunto `modelo`) e carregado na primeira foto."""
    if modelo not in TREINADO_EM:
        raise ValueError(f"modelo desconhecido: '{modelo}'. Use um de: {', '.join(TREINADO_EM)}")
    app = Flask(__name__)
    app.context_processor(lambda: {"treinado_em": TREINADO_EM[modelo]})  # o rodape de todas as respostas
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    estado = {"diag": diagnosticador}
    trava = threading.Lock()  # uma foto por vez: carregar e usar os modelos (SAM, ResNet-50) nao e seguro em paralelo

    def diag():
        if estado["diag"] is None:
            from rocole_hough_inferir import Diagnosticador
            estado["diag"] = Diagnosticador(artefato=caminho_modelo(modelo), usar_sam=usar_sam, usar_resnet=usar_resnet)
        return estado["diag"]

    def analisar(nome, dados):
        from rocole_hough_inferir import desenhar
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
        detalhe = None
        if r.get("p_resnet"):
            detalhe = {"Modelo de Hough": r["p_hough"]["rust"], "ResNet-50": r["p_resnet"]["rust"]}
        margem = r.get("margem") or 0.0
        faixa = (max(0.0, r["limiar"] - margem), min(1.0, r["limiar"] + margem)) if margem else None
        return {"nome": nome, "erro": None, "ferrugem": r["ferrugem"], "incerto": bool(r.get("incerto")), "faixa": faixa,
                "p": {NOMES[c]: v for c, v in r["p"].items()},
                "p_rust": r["p"]["rust"], "limiar": r["limiar"], "detalhe": detalhe, "n_circulos": len(r["circulos"]),
                "aviso": r["aviso"], "imagem": para_data_uri(desenhar(img, r))}

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
        return render_template("index.html", resultados=resultados, erro=None, aviso=aviso), status

    @app.errorhandler(413)
    def grande_demais(_):
        return render_template("index.html", resultados=None, aviso=None,
                               erro=f"Envio grande demais (limite de {MAX_BYTES // (1024 * 1024)} MB por vez)."), 413

    return app


def argumentos(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--porta", type=int, default=5000)
    ap.add_argument("--modelo", choices=list(TREINADO_EM), default=MODELO_PADRAO,
                    help="conjunto em que o modelo foi treinado (padrao: %(default)s); so o do rocole tem a fusao "
                         "com a ResNet-50 e a resposta 'incerto'")
    ap.add_argument("--sem-sam", action="store_true", help="contorno da folha so pela cor (mais rapido, pior no campo)")
    ap.add_argument("--sem-resnet", action="store_true", help="responde so o modelo de Hough, sem a fusao")
    return ap.parse_args(argv)


def main(argv=None):
    args = argumentos(argv)
    criar_app(usar_sam=not args.sem_sam, usar_resnet=not args.sem_resnet, modelo=args.modelo).run(
        host="127.0.0.1", port=args.porta)


if __name__ == "__main__":
    main()
