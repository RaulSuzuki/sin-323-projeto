# Relatório final: doenças do café por foto da folha

Produto final do projeto de SIN 323: o aplicativo local e os modelos que ele usa, treinados e testados no BRACOL.

> É um **apoio à decisão**. Não substitui o agrônomo.

## Resumo

- **Passo 1, saudável ou doença:** acerto de **99,0%** no teste (IC 95% 97,5% a 100%; 201 folhas).
- **Passo 2, quais doenças:** F1 macro de **0,908** (IC 95% 0,871 a 0,940) nas 180 folhas de teste com doença; as
  doenças da folha saem todas certas em 84,4% delas.
- **Severidade:** pela área da folha com cor de lesão, sem treino. Nível certo em 83,5% das folhas de validação.
- **"Incerto"** quando nem saudável nem doença passa de 60%.
- **Nenhuma doença é a principal.** O passo 1 só separa saudável de doença; no passo 2, cada doença tem o seu
  classificador e o seu limiar.
- Toda escolha foi feita em validação cruzada no treino + validação. O teste foi lido uma vez, no fim.

## Dados

**BRACOL** (Krohling, Esgario e Ventura, 2019; licença CC BY 4.0): folhas de café arábica destacadas, fotografadas
sobre fundo claro, 2048×1024 px. Cada folha tem 0 ou 1 para cada doença (bicho-mineiro, ferrugem, mancha de phoma e
cercosporiose; uma folha pode ter mais de uma) e a severidade em 5 níveis.

| | Treino | Validação | Teste |
|---|---|---|---|
| Folhas | 1.000 | 200 | 201 |
| Saudáveis | 103 | 20 | 21 |
| Com doença | 897 | 180 | 180 |
| Com mais de uma doença | 235 | 48 | 49 |
| Bicho-mineiro / ferrugem / phoma / cercosporiose | 241 / 433 / 249 / 212 | 49 / 86 / 52 / 42 | 49 / 86 / 51 / 44 |

- O arquivo do Mendeley está truncado na própria fonte: só 1.401 imagens podem ser extraídas.
- A divisão está em `outputs/hough_bracol/metadados.csv`. Imagens quase iguais (pHash) ficam sempre na mesma parte.
- A validação cruzada usa 3 dobras do treino + validação, com as quase-cópias juntas e cada combinação de doenças
  repartida entre as dobras.

## Método

1. **Contorno da folha:** o SAM (`facebook/sam-vit-base`, pré-treinado) segmenta a folha a partir de um ponto no
   centro da foto. Se não acha a folha, o contorno sai da cor.
2. **Escala:** a folha é recortada e redimensionada para que a raiz da sua área seja 700 px.
3. **Mapa de lesão:** em L\*a\*b\*, quanto cada pixel se afasta do verde da própria folha, rumo a amarelo, laranja ou
   marrom.
4. **Hough de círculos** sobre o mapa: até 60 candidatos a lesão, com raio de 4 a 48 px.
5. **Atributos:** 40 por círculo (cor, contraste, forma e textura, com LBP e GLCM) e 27 pela folha inteira.
6. **Nível 1, por círculo:** gradient boosting. Cada círculo herda a classe da sua folha (saudável ou doença).
7. **Nível 2, por folha (passo 1):** o resumo das pontuações dos círculos, com os atributos da folha, dá P(saudável)
   e P(doença).
8. **Passo 2:** um classificador por doença sobre os mesmos atributos e um nível 1 com as 5 classes (saudável e as
   quatro doenças). Uma ConvNeXt-Tiny vê a foto inteira e também dá uma probabilidade para cada doença; a resposta é a
   média das duas, com peso 0,25 para as árvores e 0,75 para a rede (escolhido na validação cruzada).
9. **Severidade:** a área com lesão, nas faixas do BRACOL.

| Parte | Arquivo | Treino |
|---|---|---|
| Passo 1: saudável ou doença | `models/bracol_hough.joblib` | nível 1 e nível 2 |
| Passo 2: quais doenças | `models/bracol_doencas.joblib` | um classificador por doença, nível 1 por doença, peso da média e limiares |
| Passo 2: quais doenças | `models/bracol_convnext.pt` | ConvNeXt-Tiny, em média com as árvores |
| Severidade | regra fixa | sem treino |

## Primeiro passo: saudável ou doença

A folha é **doença** se tem qualquer uma das quatro doenças; senão, é **saudável**. P(doença) = 1 − P(saudável), e
vale a maior das duas. Se nenhuma passa de 60%, a resposta é "Incerto", com o palpite.

**Escolha da variante.** Validação cruzada com 3 dobras; F1 macro das duas classes em cada dobra. Regra fixada antes:
uma variante só substitui a base (`completo`) com média maior **e** vitória em pelo menos 2 das 3 dobras.

| Variante | Dobra 1 | Dobra 2 | Dobra 3 | Média |
|---|---|---|---|---|
| **completo** (base: gradient boosting nos dois níveis) | 0,961 | 0,987 | 0,967 | **0,972** |
| rico (+ como são os círculos mais parecidos com doença) | 0,967 | 0,974 | 0,973 | 0,972 |
| completo, nível 2 com média de dois classificadores | 0,955 | 0,974 | 0,967 | 0,965 |
| rico, nível 2 com média de dois classificadores | 0,954 | 0,961 | 0,973 | 0,963 |
| rico, nível 2 com árvores extremamente aleatórias | 0,954 | 0,968 | 0,960 | 0,961 |
| completo, nível 2 com árvores extremamente aleatórias | 0,936 | 0,974 | 0,966 | 0,959 |
| completo, nível 2 com regressão logística | 0,932 | 0,933 | 0,954 | 0,940 |
| só a folha, sem o Hough (ablação) | 0,966 | 0,966 | 0,973 | 0,969 |
| só os círculos (ablação) | 0,925 | 0,974 | 0,954 | 0,951 |

Escolhida: **`completo`**. Nenhuma alternativa passou a regra: `rico` venceu em 2 dobras, mas empatou na média (0,9715
contra 0,9716). Sem os círculos do Hough, o F1 macro cai para 0,969; só com os círculos, para 0,951. O código
(`src/hough_treinar.py`) treina só a variante escolhida; esta comparação fica registrada em `cv_resumo.csv` e
`decisao.json`.

**Teste** (201 folhas; IC 95% por bootstrap de grupos):

| | Teste | Fora da dobra (treino + validação) |
|---|---|---|
| Acerto | **99,0%** (97,5% a 100%) | 98,9% |
| Doenças achadas | 98,9% (178 de 180; 97,2% a 100%) | 99,0% (1.066 de 1.077) |
| Saudáveis certas | 100% (21 de 21) | 98,4% (121 de 123) |
| "Incerto" | 0% | 0,1% (1 de 1.200) |

Arquivos em `outputs/hough_bracol/`: `decisao.json`, `cv_resumo.csv`, `cv_oof.csv`, `resultados_teste.json`,
`predicoes_teste.csv`, `teste_intervalos_confianca.csv` e os logs `log_cv.txt` e `log_teste.txt`.

## Segundo passo: quais doenças

Treinado nas 1.077 folhas com doença do treino + validação. Cada doença tem um classificador binário e um limiar,
escolhido pelo maior F1 fora da dobra.

**Árvores.** Validação cruzada nas mesmas 3 dobras, com a mesma regra. Sobre os atributos do passo 1, o gradient
boosting (F1 macro fora da dobra 0,745) passou as árvores extremamente aleatórias (0,728) e a regressão logística
(0,731). Depois, o **nível 1 por doença** (5 classes: saudável e as quatro doenças, treinado só com folhas de uma
doença) subiu o F1 macro para 0,825, com vitória nas 3 dobras, e foi adotado. Para cada classe entram as pontuações dos
círculos e os atributos dos 5 círculos mais parecidos com ela.

**ConvNeXt-Tiny.** A mesma receita do experimento anterior no BRACOL: pré-treinada no ImageNet, entrada 512×256 (a
foto inteira), 4 saídas multirrótulo e a severidade como tarefa auxiliar; AdamW, ciclo único de LR, até 25 épocas e
parada antecipada pelo mAP de uma validação interna (1/7 dos grupos de quem treina). Uma rede por dobra dá as
previsões fora da dobra; a rede do aplicativo é treinada no treino + validação.

**Média.** P(doença) = w · árvores + (1 − w) · ConvNeXt-Tiny, com w em 0,25, 0,5 e 0,75 e os limiares escolhidos de
novo. Mesma regra: a média só substitui as árvores com F1 macro médio maior e vitória em pelo menos 2 das 3 dobras.
A rede sozinha entra só como referência.

| F1 macro fora da dobra | Dobra 1 | Dobra 2 | Dobra 3 | Média |
|---|---|---|---|---|
| Árvores | 0,827 | 0,842 | 0,806 | 0,825 |
| **Média, peso 0,25 nas árvores** | **0,872** | **0,891** | **0,865** | **0,876** |
| Média, peso 0,5 | 0,861 | 0,885 | 0,863 | 0,870 |
| Média, peso 0,75 | 0,843 | 0,872 | 0,830 | 0,848 |
| ConvNeXt-Tiny sozinha (referência) | 0,862 | 0,863 | 0,850 | 0,858 |

A média com peso 0,25 nas árvores venceu as árvores nas 3 dobras e foi adotada. Limiares da média, escolhidos fora da
dobra: bicho-mineiro 0,66, ferrugem 0,40, mancha de phoma 0,54 e cercosporiose 0,48. Com eles, o F1 macro fora da dobra
é 0,883.

O código (`src/hough_doencas.py`) treina só a configuração adotada: gradient boosting, nível 1 por doença e média com
peso 0,25 nas árvores. A validação cruzada dele só escolhe os limiares de cada doença. As comparações acima ficam
registradas em `doencas_cv_resumo.csv`, `doencas_cv_por_doenca_*` e `doencas_fusao_*`.

**Teste** (180 folhas com doença; IC 95% por bootstrap de grupos):

| Métrica | Valor | IC 95% |
|---|---|---|
| F1 macro | **0,908** | 0,871 a 0,940 |
| Doenças da folha todas certas | 84,4% | 78,9% a 90,0% |
| A doença mais provável está entre as da folha | 97,8% | |

| Doença | Positivos | Precisão | Recall | F1 | AP |
|---|---|---|---|---|---|
| Bicho-mineiro | 49 | 0,896 | 0,878 | 0,887 | 0,941 |
| Ferrugem | 86 | 0,955 | 0,988 | 0,971 | 0,999 |
| Mancha de phoma | 51 | 0,979 | 0,902 | 0,939 | 0,991 |
| Cercosporiose | 44 | 0,809 | 0,864 | 0,835 | 0,891 |

- A cercosporiose aparece quase sempre com ferrugem (164 folhas com as duas no treino + validação, 66 só com
  ela), e as manchas das duas se confundem.
- A mina do bicho-mineiro é alongada, e o Hough procura círculos.
- A ConvNeXt-Tiny, que vê a foto inteira, ajudou mais nessas duas doenças. Fora da dobra, o F1 da cercosporiose foi
  de 0,662 (árvores) para 0,759 (média) e o do bicho-mineiro, de 0,804 para 0,852.
- Só para descrever (nada foi escolhido com isso): as árvores sozinhas tiveram F1 macro de 0,861 no mesmo teste.

Arquivos em `outputs/hough_bracol/`: `doencas_decisao.json`, `doencas_cv_*`, `convnext_oof.csv`,
`doencas_fusao_resumo.csv`, `doencas_fusao_oof.csv`, `doencas_teste.json`, `doencas_teste_por_doenca.csv`,
`doencas_predicoes_teste.csv` e os logs `log_doencas_*.txt` e `log_convnext_*.txt`.

## Severidade

Fração da folha com mapa de lesão acima de 0,30, nas faixas de área em que o BRACOL foi rotulado (Esgario et al., 2020):
saudável abaixo de 0,1%, muito baixa até 5%, baixa até 10%, alta até 15% e muito alta acima disso. Nenhum corte foi
ajustado aos dados.

| Validação do BRACOL (200 folhas) | Valor |
|---|---|
| Nível certo | 83,5% |
| Até um nível de distância | 98,0% |
| Kappa quadrático | 0,785 |
| Acerto de quem chuta sempre a classe mais comum | 61,5% |

Arquivo: `outputs/hough_bracol/severidade_validacao.json`.

## Aplicativo

`app/app.py` é uma página local (Flask). As fotos não precisam de máscara: o contorno da folha é achado na hora pelo
SAM (`src/hough_inferir.py`).

```bash
python -P app/app.py                  # depois abra http://127.0.0.1:5000
python -P app/app.py --sem-convnext   # passo 2 só com as árvores
```

- Envie até 12 fotos por vez (JPG ou PNG), com a folha no centro. A rotação do celular (EXIF) é corrigida.
- Para cada foto: saudável, doença ou incerto; se for doença, a probabilidade de cada doença, com o limiar dela, e a
  severidade estimada. Uma folha saudável mostra só o primeiro passo.
- A foto volta com o contorno da folha e os círculos do Hough, cada um na cor da classe com que mais se parece:
  azul, folha sadia; amarelo, bicho-mineiro; vermelho, ferrugem; magenta, mancha de phoma; ciano, cercosporiose.
- Um aviso aparece quando o SAM não acha a folha, quando nenhum círculo é encontrado ou quando a ConvNeXt-Tiny não
  está disponível. Com várias fotos, um resumo conta as saudáveis, as com doença (e quantas de cada doença) e as
  incertas.
- **Conferido com os modelos de verdade** em folhas de teste (telas em `docs/app/`): a 1300 (ferrugem e cercosporiose)
  saiu com as duas doenças e severidade baixa, como no rótulo; a 20 (mancha de phoma) saiu com phoma em 98% e severidade
  alta, um nível acima do rótulo; a 104 (só ferrugem, severidade muito baixa) saiu "Saudável" (85%), um dos 2 erros do
  passo 1 no teste. Em uma RTX 4050, a primeira análise levou cerca de 22 s (carrega o SAM e os modelos); depois, cerca
  de 1,3 s por foto.

## Limitações

- **Só folhas destacadas sobre fundo claro.** Folhas fotografadas na planta e de outras regiões não foram medidas.
- **Teste pequeno:** 201 folhas, 180 com doença. Os intervalos são largos.
- **O teste já tinha sido usado** pelas versões anteriores dos modelos do BRACOL (descritas em
  [RELATORIO_BRACOL.md](RELATORIO_BRACOL.md)). Nesta versão, todas as escolhas foram feitas na validação cruzada, e o
  teste foi lido uma vez.
- **Rótulo por folha, não por lesão:** cada círculo herda a classe da sua folha no treino.
- **Pouca doença passa despercebida:** as 2 folhas com doença que o passo 1 chamou de saudáveis no teste (104 e 1079)
  têm só ferrugem, com severidade muito baixa. **Cercosporiose e bicho-mineiro** continuam as doenças mais difíceis (F1
  de 0,835 e 0,887 no teste).
- **O Hough procura círculos.** Lesões grandes, alongadas ou na borda da folha são cobertas por vários círculos.
- **Severidade:** conta a área de todas as lesões juntas, sem separar por doença. Foi conferida só na validação.
- **Contorno da folha:** se a folha não está no centro da foto, ou se há várias folhas, o contorno sai errado.
- Roda na própria máquina, uma foto por vez.

## Trabalhos futuros

- Fotos do uso real: folhas fotografadas na planta, de outras lavouras e regiões.
- Bicho-mineiro: atributos de forma das manchas do mapa de lesão (alongamento, área, contorno).
- Cercosporiose: medir o centro claro e o anel escuro em volta de cada círculo; reunir mais folhas só com ela.
- Rótulo por lesão em algumas centenas de folhas, para treinar o nível 1 de verdade e dar a severidade por doença.
- Versão para celular, com um segmentador mais leve que o SAM.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `app/app.py`, `app/templates/` | aplicativo local (Flask) |
| `src/hough_inferir.py` | aplica os modelos a uma foto solta (SAM na hora) |
| `src/hough.py` | da foto aos círculos do Hough e aos atributos |
| `src/hough_treinar.py` | extração em lote, modelo de dois níveis (passo 1), validação cruzada e modelo final |
| `src/hough_doencas.py` | passo 2: limiares na validação cruzada, modelo final (média com a ConvNeXt-Tiny) e teste |
| `src/convnext_bracol.py` | ConvNeXt-Tiny: uma rede por dobra e a rede final |
| `src/hough_avaliar.py` | teste do passo 1, uma vez |
| `src/hough_dados.py` | o conjunto: fotos, classes, divisão, pastas |
| `src/folha_sam.py` | segmentação da folha com o SAM |
| `src/comum.py` | caminhos e constantes |
| `tests/` | testes automáticos (`unittest`) |
| `models/` | modelos do aplicativo |
| `outputs/hough_bracol/` | resultados de cada etapa |
| `docs/` | relatórios, resultados, telas e figuras |

Os demais arquivos de `models/` são de um experimento anterior e não são usados pelo aplicativo.

## Como reproduzir

Requer Python 3.13 e, de preferência, GPU NVIDIA. Baixe o BRACOL (<https://data.mendeley.com/datasets/yy2k5y8mxg>) e
extraia o arquivo interno em `data/bracol_raw/`. O zip está truncado na própria fonte; `tar -xf arquivo.zip` extrai as
1.401 imagens que existem. Rode da raiz do repositório (no PowerShell: `$env:PYTHONPATH = "src"`):

```bash
export PYTHONPATH=src
python -P src/hough_treinar.py mascaras --bracol   # SAM nas 1.401 folhas (GPU, ~15 min)
python -P src/hough_treinar.py extrair --bracol    # circulos do Hough e atributos
python -P src/hough_treinar.py cv --bracol         # passo 1: previsoes fora da dobra (so descrevem)
python -P src/hough_treinar.py final --bracol      # grava models/bracol_hough.joblib
python -P src/hough_avaliar.py --bracol            # teste do passo 1, uma vez
python -P src/convnext_bracol.py cv                # ConvNeXt-Tiny: uma rede por dobra (GPU, ~30 min)
python -P src/hough_doencas.py cv                  # passo 2: previsoes fora da dobra e limiar de cada doenca
python -P src/convnext_bracol.py final             # ConvNeXt-Tiny do aplicativo (GPU)
python -P src/hough_doencas.py final               # grava models/bracol_doencas.joblib
python -P src/hough_doencas.py teste               # teste do passo 2, uma vez
python -P app/app.py                               # aplicativo em http://127.0.0.1:5000
python -P -m unittest discover -s tests            # testes automaticos
```

Treinar de novo nesta máquina (RTX 4050) deu os mesmos modelos e os mesmos números; em outra GPU ou com outras versões
das bibliotecas, os números podem variar um pouco.

## Referências

- Krohling, R. A., Esgario, J. e Ventura, J. A. *BRACOL: A Brazilian Arabica Coffee Leaf images dataset*. Mendeley
  Data, 2019.
- Esgario, J. G. M., Krohling, R. A. e Ventura, J. A. *Deep learning for classification and severity estimation of
  coffee leaf biotic stress*. Computers and Electronics in Agriculture 169, 105162, 2020 (faixas de severidade).
- Kirillov, A. et al. *Segment Anything*. ICCV, 2023 (modelo `facebook/sam-vit-base`).
- Liu, Z. et al. *A ConvNet for the 2020s*. CVPR, 2022 (ConvNeXt).
- Duda, R. O. e Hart, P. E. *Use of the Hough Transformation to Detect Lines and Curves in Pictures*.
  Communications of the ACM 15(1), 1972.
- Yuen, H. K. et al. *Comparative study of Hough Transform methods for circle finding*. Image and Vision Computing
  8(1), 1990.
- Ojala, T., Pietikäinen, M. e Mäenpää, T. *Multiresolution Gray-Scale and Rotation Invariant Texture Classification
  with Local Binary Patterns*. IEEE TPAMI 24(7), 2002 (LBP).
- Haralick, R. M., Shanmugam, K. e Dinstein, I. *Textural Features for Image Classification*. IEEE Transactions on
  Systems, Man, and Cybernetics 3(6), 1973 (GLCM).
- Geurts, P., Ernst, D. e Wehenkel, L. *Extremely randomized trees*. Machine Learning 63, 2006.
