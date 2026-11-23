# Testes

Testes automáticos do código, para conferir que o aplicativo e os módulos de `src/` funcionam. Não fazem parte do
aplicativo e não têm relação com o conjunto de teste das imagens.

Usam entradas sintéticas (folhas desenhadas, conjuntos pequenos e modelos falsos), por isso não precisam das imagens
do BRACOL e do RoCoLe nem dos arquivos de `models/`.

## Como rodar

Na raiz do projeto, com o ambiente ativado:

```bash
set PYTHONPATH=src
python -P -m unittest discover -s tests
```

No Linux ou no macOS, troque a primeira linha por `export PYTHONPATH=src`. Leva de 8 a 15 minutos e termina com `OK`
quando os 245 testes passam.

Para rodar um arquivo só:

```bash
python -P -m unittest discover -s tests -p "test_app.py"
```

## O que cada arquivo confere

| Arquivo | Módulo | O que confere |
|---|---|---|
| `test_app.py` | `app/app.py` | página, envio de fotos, resposta em dois passos, "Incerto", severidade, cores dos círculos |
| `test_hough_inferir.py` | `src/hough_inferir.py` | diagnóstico de uma foto solta, contorno da folha, fusão com a ResNet-50 |
| `test_hough.py` | `src/hough.py` | mapa de lesão, círculos do Hough e atributos |
| `test_hough_treinar.py` | `src/hough_treinar.py` | modelo de dois níveis, validação cruzada e fusão |
| `test_hough_doencas.py` | `src/hough_doencas.py` | modelo das doenças; o teste é pontuado uma vez só |
| `test_hough_avaliar.py` | `src/hough_avaliar.py` | avaliação no conjunto de teste |
| `test_hough_dados.py` | `src/hough_dados.py` | conjuntos de dados, classes e divisão |
| `test_hough_recusa.py` | `src/hough_recusa.py` | faixa de recusa do detector do RoCoLe |
| `test_folha_sam.py` | `src/folha_sam.py` | segmentação da folha com o SAM |
| `test_cnn_*.py`, `test_gradcam.py` | `src/cnn_*.py`, `src/gradcam.py` | ResNet-50: modelo, treino, avaliação, validação cruzada e Grad-CAM |
| `test_preparar_rocole.py` | `src/preparar_rocole.py` | limpeza e divisão do RoCoLe |
| `test_regras.py` | `src/regras.py` | detector por regras usado como comparação |
| `test_comum.py` | `src/comum.py` | caminhos e constantes |