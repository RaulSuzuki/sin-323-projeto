# BRACOL: modelo de quatro doenças e severidade

Primeira parte do projeto de SIN 323. Uma ConvNeXt-Tiny recebe a foto de uma folha de café arábica e responde:

1. **Quais doenças aparecem**: bicho-mineiro, ferrugem, mancha de phoma e cercosporiose. Uma folha pode ter mais de uma;
2. **Qual a severidade**, em 5 níveis: saudável, muito baixa, baixa, alta, muito alta;
3. **Quão confiável é a resposta**, com opção de **recusar** ("enviar para o agrônomo") quando a confiança é baixa;
4. **Onde o modelo olhou** (mapa de calor Grad-CAM).

O aplicativo não usa este modelo. Ele usa o detector com o Hough e o modelo das doenças, descritos no
[relatório final](RELATORIO_FINAL.md). Os experimentos em fotos de campo estão no
[relatório do RoCoLe](RELATORIO_ROCOLE.md).

> É um **apoio à decisão**. Não substitui o agrônomo.

## Resultado principal

**ConvNeXt-Tiny, entrada 512×256.** Teste com 201 folhas que o modelo nunca viu:

| Métrica | Valor | IC 95% (bootstrap) |
|---|---|---|
| Acerto exato do conjunto de doenças da folha | **84,6%** | 79,6% a 89,6% |
| mAP | 0,950 | 0,917 a 0,979 |
| F1 macro | 0,904 | 0,871 a 0,935 |
| Severidade (acurácia) | 83,1% | 78,1% a 88,1% |
| Acerto nas folhas aceitas (77,6% aceitas, o resto recusado) | 92,3% | |

## Dados

- **BRACOL** (Krohling, Esgario e Ventura), licença CC BY 4.0: <https://data.mendeley.com/datasets/yy2k5y8mxg>.
  Folhas de café arábica destacadas, sobre fundo claro, fotografadas no Espírito Santo, 2048×1024 px.
  Rótulos: `miner`, `rust`, `phoma`, `cercospora` (0/1 cada) e `severity` (0 a 4).
- **O arquivo do Mendeley está truncado na própria fonte** (o SHA-256 bate com o publicado): só 1.402 das 1.747
  imagens estão nele, e uma está corrompida. Usamos **1.401 imagens**. As 345 ausentes incluem 149 folhas com
  bicho-mineiro, que pode estar sub-representado.
- A coluna `phoma` aparece no Mendeley como "brown leaf spot". Usamos o nome do CSV.
- Um conjunto anterior (Kenya, Hugging Face) foi descartado: 86% das imagens de teste tinham cópia no treino.

**Divisão 70/15/15:** 1.000 treino, 200 validação e 201 teste, estratificada pela combinação de rótulos.
Quase-duplicatas (pHash ≤ 4, 14 pares) ficam sempre no mesmo split. É a mesma divisão do detector do aplicativo
(`outputs/hough_bracol/metadados.csv`).

## Modelo

- ConvNeXt-Tiny pré-treinado no ImageNet, com duas cabeças: 4 saídas multirrótulo (`BCEWithLogitsLoss` com peso
  positivo) e 5 níveis de severidade (entropia cruzada ponderada).
- Augmentation: espelhamento, rotação e translação leves, brilho/contraste/saturação e pequenas oclusões.
- Treino: AdamW, ciclo único de LR, AMP na GPU, até 25 épocas com parada antecipada (paciência 8). A melhor época
  foi a 12 (mAP de validação 0,957). Cerca de 20 min em uma RTX 4050.
- Ajustado **só na validação**: temperatura de calibração, limiar por doença (maior F1) e limiar de incerteza para a
  recusa (aceita cerca de 80% das folhas). Tudo fica em `models/calibracao.json`.

## Resultados no teste

| Doença | Positivos | Precisão | Recall | F1 | AP |
|---|---|---|---|---|---|
| Bicho-mineiro | 49 | 0,827 | 0,878 | 0,851 | 0,934 |
| Ferrugem | 86 | 0,988 | 0,988 | 0,988 | 1,000 |
| Mancha de Phoma | 51 | 0,979 | 0,922 | 0,949 | 0,990 |
| Cercosporiose | 44 | 0,745 | 0,932 | 0,828 | 0,875 |

- Acerto exato em folhas com 2 ou mais doenças (49 no teste): 67,3%.
- Folhas saudáveis (21): todas identificadas, sem falso positivo.
- Severidade: erro médio de 0,17 nível, 100% dentro de 1 nível, kappa quadrático 0,885.
- Calibração: ECE 0,019 antes e 0,021 depois da temperatura.
- **Recusa:** 19 dos 31 erros (61%) foram recusados, contra 26 das 170 folhas corretas (15%).
- **Erros:** 18 com uma doença a mais, 7 com doença faltando e 6 com os dois. As folhas com várias doenças concentram
  os erros (33% de erro, contra 10% nas folhas com uma só doença).
- **Grad-CAM:** 74,6% da atenção cai sobre a folha, que ocupa 32,6% da imagem (medida grosseira: não há máscaras de
  lesão).

### Robustez (as mesmas 201 folhas, degradadas)

| Perturbação | Acerto exato | Taxa de recusa |
|---|---|---|
| original | 83,1% | 20,4% |
| brilho ×0,5 / ×1,6 | 81,6% / 81,1% | 23,9% / 21,9% |
| contraste ×0,5 | 83,6% | 24,9% |
| zoom 60% | 79,6% | 26,9% |
| fundo de campo sintético | 76,1% | 27,4% |
| ruído gaussiano | 75,6% | 29,4% |
| desfoque r=2 / r=4 | 73,6% / 50,7% | 36,8% / 52,7% |
| JPEG qualidade 15 | 69,7% | 37,8% |
| rotação 90° | 69,2% | 33,8% |
| baixa resolução (1/4) | 67,2% | 39,3% |

O modelo aguenta variações de luz, mas é sensível a desfoque, compressão, baixa resolução e rotação. A recusa sobe
junto com a degradação. A linha "original" fica em 83,1%, e não 84,6%, porque este teste roda sem TTA.

## Outras abordagens testadas

Regra fixada antes de rodar o teste: um modelo novo só substitui o atual se for **melhor na validação**. Foram
testados resolução maior (768×384), augmentation de degradação, recorte da folha, validação cruzada em 3 dobras,
outras arquiteturas (ResNet-50, ConvNeXt-Small) e conjuntos de modelos.

| Modelo (acerto exato) | Validação (200) | Teste (201) | Severidade no teste |
|---|---|---|---|
| **ConvNeXt-Tiny 512×256 (escolhido)** | **85,0%** | **84,6%** | 83,1% |
| ConvNeXt-Tiny 768×384, 1 modelo | 87,5% | 84,1% | 86,1% |
| ConvNeXt-Tiny 768×384 com recorte, 1 modelo | 88,5% | 81,6% | 81,1% |
| ConvNeXt-Tiny 768×384, 3 modelos | 85,0% | 85,6% | 89,6% |
| ConvNeXt-Tiny 768×384, 3 modelos, 25 épocas | 84,5% | 86,6% | 89,1% |
| ConvNeXt-Tiny 768×384 com recorte, 3 modelos | 84,0% | 87,1% | 90,0% |
| ConvNeXt-Small 768×384, 3 modelos | 86,5% | 86,1% | 88,1% |
| ResNet-50 768×384, 3 modelos | 81,0% | 82,6% | 89,6% |
| Conjunto de 3 arquiteturas | 86,0% | 84,6% | 89,6% |

- Nenhuma alternativa ficou acima na validação de forma consistente. O melhor modelo na validação (88,5%) foi o
  **pior** no teste (81,6%). Com 200 folhas, uma folha vale 0,5 ponto.
- As diferenças no teste ficam dentro do intervalo do modelo escolhido (79,6% a 89,6%). Escolher pelo teste o
  invalidaria como medida.
- O escolhido é **um modelo só, com entrada menor**: mais rápido e leve que 3 modelos em 768×384.
- Os modelos 768×384 erram menos a severidade no teste (cerca de 89% contra 83%); na validação a diferença é menor.

## Limitações

- **Fotos de campo:** o BRACOL tem folha destacada sobre fundo claro. Com a folha na planta, o desempenho não foi
  medido.
- Conjunto truncado e pequeno (1.401 folhas), de uma região e de um conjunto de câmeras.
- A severidade vem dos níveis rotulados, não de segmentação de lesão por pixel.
- Teste com 201 folhas: intervalos largos, principalmente nas folhas com várias doenças.
- O modelo analisa uma folha por vez; uma folha não representa a planta inteira.

## Estrutura

Caminhos do projeto original, onde ficam o código, os pesos e os resultados deste modelo. Como o aplicativo não usa
este modelo, esses arquivos não fazem parte desta pasta final.

| Caminho | Conteúdo |
|---|---|
| `src/comum.py` | caminhos e constantes |
| `src/auditar_bracol.py`, `src/preparar_dados.py` | auditoria do conjunto, limpeza, cache 512×256, quase-duplicatas, divisão |
| `src/modelo.py`, `src/treinar.py` | dataset, augmentation, ConvNeXt multitarefa, treino |
| `src/avaliar.py`, `src/gradcam.py` | calibração, métricas, recusa, Grad-CAM |
| `src/intervalos.py`, `src/robustez.py`, `src/erros.py` | intervalos de confiança, robustez, análise de erros |
| `models/melhor_modelo.pt` | pesos do modelo |
| `models/calibracao.json` | temperaturas, limiares por doença e limiar de recusa |
| `outputs/` | métricas, figuras, predições do teste e `comparacao_modelos.csv` |

Para usar o modelo em outro programa: `carregar()` em `src/modelo.py` carrega os pesos. A imagem é redimensionada
para 512×256 e normalizada com a média e o desvio do ImageNet (`src/comum.py`); depois aplicam-se a temperatura, os
limiares por doença e o limiar de recusa de `models/calibracao.json`, como em `src/avaliar.py`.

## Como reproduzir

Requer Python 3.13 e, de preferência, GPU NVIDIA. Os comandos deste modelo rodam no projeto original.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Baixe o BRACOL (link acima) e extraia o arquivo interno para `data/bracol_raw/`. O zip interno está truncado; o
`tar` do Windows extrai o que existe: `tar -xf arquivo.zip`.

```bash
export PYTHONPATH=src
python -P src/preparar_dados.py          # valida, cria o cache 512x256 e divide
python -P src/treinar.py                 # treino; grava models/melhor_modelo.pt (sobrescreve o atual)
python -P src/avaliar.py                 # calibra na validacao e avalia no teste
python -P src/intervalos.py              # intervalos de confianca
python -P src/robustez.py                # robustez
python -P src/erros.py                   # analise de erros
```

Para usar os pesos já treinados, coloque `melhor_modelo.pt` em `models/` e pule o treino. Treinar de novo na GPU dá
números próximos, mas não idênticos.

## Referência

Krohling, R. A.; Esgario, J. G. M.; Ventura, J. A. *BRACOL – a Brazilian Arabica coffee leaf image dataset
for identifying and quantifying coffee diseases and pests*. Mendeley Data, 2019. CC BY 4.0.
