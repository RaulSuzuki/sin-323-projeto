# Testes

Testes automáticos do código, para conferir que o aplicativo e os módulos de `src/` funcionam. Não fazem parte do
aplicativo e não têm relação com o conjunto de teste das imagens.

Usam entradas sintéticas (folhas desenhadas, conjuntos pequenos e modelos falsos), por isso não precisam das imagens
nem dos arquivos de `models/`.

## Como rodar

Na raiz do projeto, com o ambiente ativado:

```bash
set PYTHONPATH=src
python -P -m unittest discover -s tests
```

No Linux ou no macOS, troque a primeira linha por `export PYTHONPATH=src`. Leva cerca de 10 minutos e termina com `OK`
quando os 148 testes passam.

Para rodar um arquivo só:

```bash
python -P -m unittest discover -s tests -p "test_app.py"
```

## O que cada arquivo confere

| Arquivo | Módulo | O que confere |
|---|---|---|
| `test_app.py` | `app/app.py` | página, envio de fotos, resposta em dois passos, "Incerto", severidade, cores dos círculos |
| `test_hough_inferir.py` | `src/hough_inferir.py` | diagnóstico de uma foto solta, contorno da folha |
| `test_hough.py` | `src/hough.py` | mapa de lesão, círculos do Hough e atributos |
| `test_hough_treinar.py` | `src/hough_treinar.py` | modelo de dois níveis (saudável ou doença) e validação cruzada |
| `test_hough_doencas.py` | `src/hough_doencas.py` | modelo das doenças, média com a ConvNeXt-Tiny; o teste é pontuado uma vez só |
| `test_convnext_bracol.py` | `src/convnext_bracol.py` | ConvNeXt-Tiny: entrada, rede e previsões fora da dobra |
| `test_hough_avaliar.py` | `src/hough_avaliar.py` | avaliação no conjunto de teste |
| `test_hough_dados.py` | `src/hough_dados.py` | conjuntos de dados, classes e divisão |
| `test_folha_sam.py` | `src/folha_sam.py` | segmentação da folha com o SAM |
| `test_comum.py` | `src/comum.py` | caminhos e constantes |