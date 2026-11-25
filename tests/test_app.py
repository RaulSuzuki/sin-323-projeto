"""Testes do aplicativo (app/app.py) com um diagnosticador falso: a pagina, o envio de fotos e a leitura de imagens."""
import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import app as aplicativo  # noqa: E402


SIM = '<span class="veredito c-doenca">Doença</span>'  # o veredito, nao o rotulo da barra de probabilidade
SAUDAVEL = '<span class="veredito c-healthy">Saudável</span>'


TALVEZ = '<span class="veredito talvez">Incerto</span>'


class DiagFalso:
    def __init__(self, doenca=True, p=None, extra=None):
        self.doenca, self.p, self.extra, self.formas = doenca, p, extra or {}, []

    def diagnosticar(self, img):
        self.formas.append(img.shape)
        p = self.p or ({"healthy": 0.3, "disease": 0.7} if self.doenca else {"healthy": 0.8, "disease": 0.2})
        r = {"positivo": self.doenca, "p": p, "limiar": 0.5,
             "circulos": [(50.0, 40.0, 10.0, 0.9)], "mascara": np.ones(img.shape[:2], bool), "confianca": 0.8,
             "aviso": "aviso de teste"}
        r.update(self.extra)
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

    def test_foto_com_doenca_mostra_veredito_probabilidades_e_imagem(self):
        r = self._enviar([(jpeg(), "folha1.jpg")])
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("folha1.jpg", html)
        self.assertIn(SIM, html)
        self.assertNotIn(SAUDAVEL, html)
        self.assertIn("70%", html)
        self.assertIn("data:image/jpeg;base64,", html)
        self.assertIn("aviso de teste", html)
        self.assertEqual(self.diag.formas, [(90, 160, 3)])

    def test_foto_saudavel(self):
        cliente = aplicativo.criar_app(DiagFalso(doenca=False)).test_client()
        r = cliente.post("/", data={"fotos": [(io.BytesIO(jpeg()), "ok.jpg")]}, content_type="multipart/form-data")
        html = r.get_data(as_text=True)
        self.assertIn(SAUDAVEL, html)
        self.assertNotIn(SIM, html)

    def test_foto_incerta_mostra_incerto_e_manda_ao_agronomo(self):
        """Nem saudavel (45%) nem doenca (55%) passa de 60%: a resposta e 'incerto', com o palpite."""
        p = {"healthy": 0.45, "disease": 0.55}
        cliente = aplicativo.criar_app(DiagFalso(p=p)).test_client()
        r = cliente.post("/", data={"fotos": [(io.BytesIO(jpeg()), "duvida.jpg")]}, content_type="multipart/form-data")
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn(TALVEZ, html)
        self.assertNotIn(SIM, html)
        self.assertNotIn(SAUDAVEL, html)
        self.assertIn("mostre a folha a um agrônomo", html)
        self.assertIn("Palpite do modelo: <strong>Doença</strong>", html)
        self.assertIn("nem saudável nem doença passou de 60%", html)

    def test_foto_segura_nao_e_marcada_como_incerta(self):
        cliente = aplicativo.criar_app(DiagFalso(doenca=True)).test_client()
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

    def _html(self, diag, n=1):
        cliente = aplicativo.criar_app(diag).test_client()
        r = cliente.post("/", data={"fotos": [(io.BytesIO(jpeg()), f"f{i}.jpg") for i in range(n)]},
                         content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        return r.get_data(as_text=True)

    TIPOS = 'aria-label="Doenças"'  # a parte com a probabilidade de cada classe

    def test_saudavel_mostra_so_saudavel_e_doenca(self):
        """Saudavel: sem a parte das doencas nem a severidade."""
        html = self._html(DiagFalso(p={"healthy": 0.65, "disease": 0.35}, extra={"area_lesao": 0.07}))
        self.assertIn(SAUDAVEL, html)
        self.assertNotIn(SIM, html)
        self.assertIn("<span>Doença</span><span>35%</span>", html)
        self.assertNotIn(self.TIPOS, html)
        self.assertNotIn("Severidade estimada</span>", html)

    def test_doenca_sem_o_modelo_das_doencas_avisa(self):
        html = self._html(DiagFalso(p={"healthy": 0.10, "disease": 0.90}))
        self.assertIn(SIM, html)
        self.assertIn("<span>Doença</span><span>90%</span>", html)
        tipos = html.split(self.TIPOS)[1]
        self.assertIn("Modelo das doenças indisponível", tipos)
        self.assertNotIn("Doença mais provável", tipos)

    def test_classificar(self):
        k = aplicativo.classificar({"healthy": 0.2, "disease": 0.8})
        self.assertEqual((k["doenca"], k["incerto"]), (True, False))
        self.assertAlmostEqual(k["p_doenca"], 0.8)
        k = aplicativo.classificar({"healthy": 0.55, "disease": 0.45})
        self.assertEqual((k["doenca"], k["incerto"]), (False, True))

    def test_resumo_conta_saudaveis_doencas_e_tipos(self):
        class Varios(DiagFalso):
            ps = [{"healthy": 0.8, "disease": 0.2}, {"healthy": 0.1, "disease": 0.9}, {"healthy": 0.5, "disease": 0.5}]

            def diagnosticar(self, img):
                self.p = self.ps[len(self.formas) % 3]
                return super().diagnosticar(img)

        html = self._html(Varios(extra={"p_doencas": self.P_DOENCAS, "limiares_doencas": self.LIMIARES}), n=3)
        resumo = html.split('<dl class="resumo">')[1].split("</dl>")[0]
        self.assertIn("<dd>1</dd><dt>saudável</dt>", resumo)
        self.assertIn("<dd>1</dd><dt>com doença (1 bicho-mineiro, 1 ferrugem, 0 mancha de phoma, 0 cercosporiose)</dt>",
                      resumo)
        self.assertIn("<dd>1</dd><dt>incerta</dt>", resumo)

    P_DOENCAS = {"miner": 0.62, "rust": 0.81, "phoma": 0.05, "cercospora": 0.20}
    LIMIARES = {"miner": 0.35, "rust": 0.70, "phoma": 0.33, "cercospora": 0.48}

    def _com_doencas(self, p_doencas=None, extra=None):
        return DiagFalso(p={"healthy": 0.05, "disease": 0.95},
                         extra={"p_doencas": p_doencas or self.P_DOENCAS, "limiares_doencas": self.LIMIARES,
                                **(extra or {})})

    def test_com_o_modelo_das_doencas_cada_doenca_tem_a_sua_probabilidade(self):
        """Multirrotulo: bicho-mineiro e ferrugem passam dos seus limiares; as outras duas aparecem, sem passar."""
        html = self._html(self._com_doencas())
        tipos = html.split(self.TIPOS)[1]
        self.assertIn("Doenças mais prováveis</span><strong>Bicho-mineiro + Ferrugem</strong>", tipos)
        for nome, p in (("Bicho-mineiro", "62%"), ("Ferrugem", "81%"), ("Mancha de phoma", "5%"), ("Cercosporiose", "20%")):
            self.assertIn(f"<span>{nome}</span><span>{p}</span>", tipos)
        self.assertIn('style="left: 70.0%"', tipos)  # o traco do limiar da ferrugem
        self.assertNotIn("ConvNeXt", tipos)

    def test_com_a_convnext_a_pagina_diz_que_e_uma_media(self):
        html = self._html(self._com_doencas(extra={"p_doencas_convnext": {"miner": 0.5}}))
        self.assertIn("média das árvores (círculos) com a ConvNeXt-Tiny", html.split(self.TIPOS)[1])

    def test_nenhuma_doenca_passa_do_limiar_mostra_a_mais_provavel(self):
        html = self._html(self._com_doencas({"miner": 0.10, "rust": 0.30, "phoma": 0.20, "cercospora": 0.40}))
        self.assertIn("Doença mais provável</span><strong>Cercosporiose</strong>", html.split(self.TIPOS)[1])

    def test_resumo_conta_cada_doenca_do_modelo_das_doencas(self):
        html = self._html(self._com_doencas(), n=2)
        resumo = html.split('<dl class="resumo">')[1].split("</dl>")[0]
        self.assertIn("<dd>2</dd><dt>com doença (2 bicho-mineiro, 2 ferrugem, 0 mancha de phoma, 0 cercosporiose)</dt>",
                      resumo)

    def test_resumo_sem_o_modelo_das_doencas_nao_lista_doencas(self):
        html = self._html(DiagFalso(), n=2)
        self.assertIn("<dd>2</dd><dt>com doença</dt>", html.split('<dl class="resumo">')[1])

    def test_por_doenca(self):
        linhas, provaveis = aplicativo.por_doenca(self.P_DOENCAS, self.LIMIARES)
        self.assertEqual(provaveis, ["Bicho-mineiro", "Ferrugem"])
        self.assertEqual([x["provavel"] for x in linhas], [True, True, False, False])

    def test_severidade_pela_area_com_lesao(self):
        html = self._html(DiagFalso(extra={"area_lesao": 0.07}))
        self.assertIn("Severidade estimada</span><strong>Baixa</strong>", html)
        self.assertIn("7,0% da folha com cor de lesão", html)
        self.assertNotIn("Severidade estimada</span>", self._html(DiagFalso()))  # sem a area, nao ha severidade

    def test_circulo_na_cor_da_classe_com_que_mais_se_parece(self):
        img = np.zeros((100, 160, 3), np.uint8)
        r = {"mascara": np.ones((100, 160), bool), "circulos": [(80.0, 50.0, 20.0, 0.1)],
             "circulos_classes": [{"healthy": 0.2, "disease": 0.8}]}
        saida = aplicativo.desenhar_classes(img, r)
        self.assertEqual(tuple(int(v) for v in saida[50, 80 + 20 + 2]), aplicativo.CORES_CIRCULO["disease"])
        r.pop("circulos_classes")  # sem probabilidades por classe: cinza
        self.assertEqual(tuple(int(v) for v in aplicativo.desenhar_classes(img, r)[50, 102]), aplicativo.CINZA)

    def test_com_o_modelo_das_doencas_cada_circulo_na_cor_da_sua_doenca(self):
        """circulos_doencas (saudavel e as quatro doencas) vale mais que as classes do modelo de Hough."""
        img = np.zeros((100, 160, 3), np.uint8)
        r = {"mascara": np.ones((100, 160), bool), "circulos": [(80.0, 50.0, 20.0, 0.1)],
             "circulos_classes": [{"healthy": 0.2, "disease": 0.8}],
             "circulos_doencas": [{"healthy": 0.1, "miner": 0.1, "rust": 0.1, "phoma": 0.6, "cercospora": 0.1}]}
        saida = aplicativo.desenhar_classes(img, r)
        self.assertEqual(tuple(int(v) for v in saida[50, 102]), aplicativo.CORES_CIRCULO["phoma"])

    def test_cada_doenca_conhecida_tem_uma_cor_propria(self):
        for classes in (["healthy", "miner", "rust", "phoma", "cercospora"], ["healthy", "disease"]):
            cores = [aplicativo.CORES_CIRCULO[c] for c in classes]
            self.assertEqual(len(set(cores)), len(classes), classes)
        self.assertEqual(aplicativo.cor_css("rust"), "#e62828")

    def test_legenda_com_as_doencas_do_modelo_das_doencas(self):
        circ = [{"healthy": 0.2, "miner": 0.2, "rust": 0.2, "phoma": 0.2, "cercospora": 0.2}]
        diag = self._com_doencas()
        diag.extra["circulos_doencas"] = circ
        chaves = self._html(diag).split('<div class="chaves"')[1].split("</div>")[0]
        for nome, classe in (("saudável", "healthy"), ("bicho-mineiro", "miner"), ("ferrugem", "rust"),
                             ("mancha de phoma", "phoma"), ("cercosporiose", "cercospora")):
            self.assertIn(f'<i class="ponto" style="border-color: {aplicativo.cor_css(classe)}"></i>{nome}</span>',
                          chaves)
        # sem a pontuacao dos circulos por doenca, a legenda e a das classes do passo 1
        chaves = self._html(self._com_doencas()).split('<div class="chaves"')[1].split("</div>")[0]
        self.assertIn("doença</span>", chaves)
        self.assertNotIn("cercosporiose", chaves)

    def test_so_os_modelos_do_bracol(self):
        import hough_dados as hd
        self.assertEqual(aplicativo.ARTEFATO, hd.carregar_conjunto("bracol").artefato)
        self.assertEqual(aplicativo.ARTEFATO_DOENCAS.name, "bracol_doencas.joblib")
        self.assertFalse(aplicativo.argumentos([]).sem_convnext)
        self.assertTrue(aplicativo.argumentos(["--sem-convnext"]).sem_convnext)
        with self.assertRaises(SystemExit):
            aplicativo.argumentos(["--modelo", "outro"])  # o aplicativo so tem os modelos do BRACOL

    def test_rodape_diz_em_que_fotos_os_modelos_foram_treinados(self):
        padrao = self.cliente.get("/").get_data(as_text=True)
        self.assertIn("BRACOL", padrao)
        self.assertIn("arábica", padrao)
        self.assertIn("Não substitui o agrônomo", padrao)

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
