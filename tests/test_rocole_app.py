"""Testes do aplicativo (app/app.py) com um diagnosticador falso: a pagina, o envio de fotos e a leitura de imagens."""
import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import app as aplicativo  # noqa: E402


SIM = '<span class="veredito sim">Ferrugem</span>'  # o veredito, nao o rotulo da barra de probabilidade
NAO = '<span class="veredito nao">Sem ferrugem</span>'


TALVEZ = '<span class="veredito talvez">Incerto</span>'


class DiagFalso:
    def __init__(self, ferrugem=True, incerto=None):
        self.ferrugem, self.incerto, self.formas = ferrugem, incerto, []

    def diagnosticar(self, img):
        self.formas.append(img.shape)
        p = {"healthy": 0.1, "red_spider_mite": 0.2, "rust": 0.7} if self.ferrugem else \
            {"healthy": 0.8, "red_spider_mite": 0.15, "rust": 0.05}
        r = {"ferrugem": self.ferrugem, "p": p, "p_hough": p, "p_resnet": None, "limiar": 0.3,
             "circulos": [(50.0, 40.0, 10.0, 0.9)], "mascara": np.ones(img.shape[:2], bool), "confianca": 0.8,
             "aviso": "aviso de teste"}
        if self.incerto is not None:  # diagnosticadores antigos nao tem a chave: a pagina tem de funcionar sem ela
            r.update({"incerto": self.incerto, "margem": 0.15})
        return r


def jpeg(largura=160, altura=90, orientacao=None):
    im = Image.new("RGB", (largura, altura), (40, 150, 40))
    buf = io.BytesIO()
    if orientacao:
        exif = Image.Exif()
        exif[0x0112] = orientacao
        im.save(buf, "JPEG", exif=exif)
    else:
        im.save(buf, "JPEG")
    return buf.getvalue()


def png(modo, tamanho=(120, 80)):
    buf = io.BytesIO()
    Image.new(modo, tamanho).save(buf, "PNG")
    return buf.getvalue()


class TestLerImagem(unittest.TestCase):
    def test_jpeg_comum_vira_bgr(self):
        img = aplicativo.ler_imagem(jpeg())
        self.assertEqual(img.shape, (90, 160, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertGreater(int(img[45, 80, 1]), int(img[45, 80, 0]))  # verde

    def test_foto_de_celular_girada_pelo_exif_fica_em_pe(self):
        img = aplicativo.ler_imagem(jpeg(200, 100, orientacao=6))  # 6 = girar 90 graus
        self.assertEqual(img.shape, (200, 100, 3))

    def test_png_com_transparencia_e_em_tons_de_cinza_viram_3_canais(self):
        self.assertEqual(aplicativo.ler_imagem(png("RGBA")).shape, (80, 120, 3))
        self.assertEqual(aplicativo.ler_imagem(png("L")).shape, (80, 120, 3))

    def test_arquivo_que_nao_e_imagem_levanta_value_error(self):
        for ruim in (b"isto e um texto, nao uma foto", b"", b"\xff\xd8\xff lixo"):
            with self.assertRaises(ValueError):
                aplicativo.ler_imagem(ruim)


class TestPagina(unittest.TestCase):
    def setUp(self):
        self.diag = DiagFalso()
        self.cliente = aplicativo.criar_app(self.diag).test_client()

    def _enviar(self, arquivos):
        return self.cliente.post("/", data={"fotos": [(io.BytesIO(d), n) for d, n in arquivos]},
                                 content_type="multipart/form-data")

    def test_pagina_inicial_tem_o_formulario(self):
        r = self.cliente.get("/")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn('name="fotos"', html)
        self.assertIn("multiple", html)
        self.assertIn("Não substitui o agrônomo", html)

    def test_foto_com_ferrugem_mostra_veredito_probabilidades_e_imagem(self):
        r = self._enviar([(jpeg(), "folha1.jpg")])
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("folha1.jpg", html)
        self.assertIn(SIM, html)
        self.assertNotIn(NAO, html)
        self.assertIn("70%", html)
        self.assertIn("data:image/jpeg;base64,", html)
        self.assertIn("aviso de teste", html)
        self.assertEqual(self.diag.formas, [(90, 160, 3)])

    def test_foto_sem_ferrugem(self):
        cliente = aplicativo.criar_app(DiagFalso(ferrugem=False)).test_client()
        r = cliente.post("/", data={"fotos": [(io.BytesIO(jpeg()), "ok.jpg")]}, content_type="multipart/form-data")
        html = r.get_data(as_text=True)
        self.assertIn(NAO, html)
        self.assertNotIn(SIM, html)

    def test_foto_incerta_mostra_incerto_e_manda_ao_agronomo(self):
        cliente = aplicativo.criar_app(DiagFalso(ferrugem=True, incerto=True)).test_client()
        r = cliente.post("/", data={"fotos": [(io.BytesIO(jpeg()), "duvida.jpg")]}, content_type="multipart/form-data")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn(TALVEZ, html)
        self.assertNotIn(SIM, html)
        self.assertNotIn(NAO, html)
        self.assertIn("mostre a folha a um agrônomo", html)
        self.assertIn("15% a 45%", html)  # a faixa de recusa: limiar 30% mais ou menos 15

    def test_foto_segura_nao_e_marcada_como_incerta(self):
        cliente = aplicativo.criar_app(DiagFalso(ferrugem=True, incerto=False)).test_client()
        r = cliente.post("/", data={"fotos": [(io.BytesIO(jpeg()), "certa.jpg")]}, content_type="multipart/form-data")
        html = r.get_data(as_text=True)
        self.assertIn(SIM, html)
        self.assertNotIn(TALVEZ, html)

    def test_varias_fotos_de_uma_vez(self):
        r = self._enviar([(jpeg(), "a.jpg"), (jpeg(200, 100, 6), "b.jpg")])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.diag.formas, [(90, 160, 3), (200, 100, 3)])

    def test_sem_arquivo_e_erro_400_com_mensagem(self):
        for r in (self.cliente.post("/", data={}, content_type="multipart/form-data"), self._enviar([(b"", "")])):
            self.assertEqual(r.status_code, 400)
            self.assertIn("Escolha pelo menos uma foto", r.get_data(as_text=True))

    def test_arquivo_de_texto_e_erro_400_com_mensagem_e_sem_traceback(self):
        r = self._enviar([(b"nao sou uma foto", "notas.txt")])
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 400)
        self.assertIn("notas.txt", html)
        self.assertIn("não é uma imagem", html)
        self.assertNotIn("Traceback", html)
        self.assertEqual(self.diag.formas, [])

    def test_uma_foto_boa_e_um_arquivo_ruim_mostra_os_dois(self):
        r = self._enviar([(jpeg(), "boa.jpg"), (b"lixo", "ruim.bin")])
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn(SIM, html)
        self.assertIn("ruim.bin", html)
        self.assertIn("não é uma imagem", html)

    def test_erro_do_modelo_vira_mensagem(self):
        class Quebrado:
            def diagnosticar(self, img):
                raise ValueError("imagem invalida: precisa ser colorida e ter pelo menos 32 px de lado")

        cliente = aplicativo.criar_app(Quebrado()).test_client()
        r = cliente.post("/", data={"fotos": [(io.BytesIO(jpeg(20, 20)), "mini.jpg")]}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertIn("pelo menos 32 px", r.get_data(as_text=True))

    def test_dois_envios_ao_mesmo_tempo_nao_entram_juntos_no_modelo(self):
        """Dois cliques seguidos em 'Analisar' chegavam juntos ao modelo e derrubavam a carga do SAM (erro 500)."""
        import threading
        import time

        class Lento(DiagFalso):
            dentro = maximo = 0

            def diagnosticar(self, img):
                Lento.dentro += 1
                Lento.maximo = max(Lento.maximo, Lento.dentro)
                time.sleep(0.15)
                Lento.dentro -= 1
                return super().diagnosticar(img)

        app = aplicativo.criar_app(Lento())
        status = []

        def enviar():
            r = app.test_client().post("/", data={"fotos": [(io.BytesIO(jpeg()), "a.jpg")]},
                                       content_type="multipart/form-data")
            status.append(r.status_code)

        fios = [threading.Thread(target=enviar) for _ in range(3)]
        for f in fios:
            f.start()
        for f in fios:
            f.join()
        self.assertEqual(status, [200, 200, 200])
        self.assertEqual(Lento.maximo, 1)

    def test_falha_interna_vira_mensagem_e_status_500_sem_traceback(self):
        class Quebrado:
            def diagnosticar(self, img):
                raise RuntimeError("CUDA out of memory")

        cliente = aplicativo.criar_app(Quebrado()).test_client()
        r = cliente.post("/", data={"fotos": [(io.BytesIO(jpeg()), "foto.jpg")]}, content_type="multipart/form-data")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 500)
        self.assertIn("foto.jpg", html)
        self.assertIn("falha interna", html)
        self.assertNotIn("Traceback", html)
        self.assertNotIn("CUDA", html)

    def test_modelo_de_outro_conjunto_mostra_outra_lesao(self):
        class Geral(DiagFalso):
            def diagnosticar(self, img):
                r = super().diagnosticar(img)
                r["p"] = {"healthy": 0.1, "other": 0.2, "rust": 0.7}
                return r

        cliente = aplicativo.criar_app(Geral()).test_client()
        r = cliente.post("/", data={"fotos": [(io.BytesIO(jpeg()), "a.jpg")]}, content_type="multipart/form-data")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("Outra lesão", html)
        self.assertNotIn("Ácaro-vermelho", html)

    def test_caminho_do_modelo_de_cada_conjunto(self):
        import rocole_hough_dados as hd
        for nome in ("rocole", "bracol"):
            self.assertEqual(aplicativo.caminho_modelo(nome), hd.carregar_conjunto(nome).artefato)

    def test_o_modelo_padrao_e_o_do_bracol(self):
        self.assertEqual(aplicativo.MODELO_PADRAO, "bracol")
        self.assertEqual(aplicativo.argumentos([]).modelo, "bracol")
        self.assertEqual(aplicativo.argumentos(["--modelo", "rocole"]).modelo, "rocole")
        with self.assertRaises(SystemExit):
            aplicativo.argumentos(["--modelo", "ambos"])  # nao ha modelo dos dois conjuntos juntos

    def test_rodape_diz_em_que_fotos_o_modelo_carregado_foi_treinado(self):
        padrao = self.cliente.get("/").get_data(as_text=True)
        self.assertIn("BRACOL", padrao)
        self.assertIn("arábica", padrao)
        self.assertNotIn("RoCoLe", padrao)
        rocole = aplicativo.criar_app(DiagFalso(), modelo="rocole").test_client().get("/").get_data(as_text=True)
        self.assertIn("RoCoLe", rocole)
        self.assertIn("robusta", rocole)
        self.assertNotIn("BRACOL", rocole)
        for html in (padrao, rocole):
            self.assertIn("Não substitui o agrônomo", html)

    def test_rodape_aparece_tambem_depois_de_um_envio_e_num_erro(self):
        self.assertIn("BRACOL", self._enviar([(jpeg(), "a.jpg")]).get_data(as_text=True))
        self.assertIn("BRACOL", self.cliente.post("/", data={}, content_type="multipart/form-data").get_data(as_text=True))

    def test_nome_do_arquivo_e_escapado(self):
        r = self._enviar([(jpeg(), "<script>alert(1)</script>.jpg")])
        html = r.get_data(as_text=True)
        self.assertNotIn("<script>alert(1)", html)
        self.assertIn("&lt;script&gt;", html)

    def test_limite_de_fotos_por_envio(self):
        r = self._enviar([(jpeg(), f"f{i}.jpg") for i in range(aplicativo.MAX_FOTOS + 2)])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(self.diag.formas), aplicativo.MAX_FOTOS)
        self.assertIn(f"só as {aplicativo.MAX_FOTOS} primeiras", r.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
