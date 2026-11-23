# Relatório final: doenças do café por foto da folha

Produto final do projeto de SIN 323: o aplicativo local e os modelos que ele usa. Os experimentos que levaram a ele
estão no [relatório do RoCoLe](RELATORIO_ROCOLE.md).

> É um **apoio à decisão**. Não substitui o agrônomo.

## Resumo

- **Saudável ou doença:** acerto de **99,5%** no teste do BRACOL (IC 95% 98,5% a 100%; 201 folhas) e de **96,6%** no
  teste do RoCoLe, com `--modelo rocole` (IC 95% 92,3% a 99,1%; 208 fotos).
- **Quais doenças** (BRACOL): F1 macro **0,860** (IC 95% 0,815 a 0,899) nas 180 folhas de teste com doença; o conjunto
  exato de doenças sai certo em 78,3% delas. Bicho-mineiro (F1 0,809) e cercosporiose (0,720) são as mais difíceis.
- **Severidade:** estimada pela área da folha com lesão, sem treino. Nível certo em 83,5% das folhas de validação.
- **"Incerto"** quando nem saudável nem doença passa de 60%.
- Toda escolha foi feita em validação cruzada no treino + validação. O teste foi pontuado uma vez por modelo.

## Dados

| | BRACOL (modelo padrão) | RoCoLe (`--modelo rocole`) |
|---|---|---|
| Espécie e foto | café arábica; folha destacada, fundo claro | café robusta; folha na planta, fundo de campo |
| Rótulos | bicho-mineiro, ferrugem, mancha de phoma e cercosporiose (uma folha pode ter mais de uma) e severidade em 5 níveis | saudável, ácaro-vermelho ou ferrugem |
| Imagens | 1.401 folhas | 1.479 fotos |
| Treino / validação / teste | 1.000 / 200 / 201 | 1.061 / 210 / 208 |
| Divisão | `outputs/hough_bracol/metadados.csv` | `outputs/metadados.csv` |

Imagens quase iguais (pHash) e, no RoCoLe, as fotos da mesma planta ficam sempre na mesma parte da divisão.

## Método

1. **Contorno da folha:** o SAM segmenta a folha a partir de um ponto no centro da foto. Se não acha a folha, o
   contorno sai da cor.
2. **Escala:** a folha é recortada e redimensionada para que a raiz da sua área seja 700 px.
3. **Mapa de lesão:** em L\*a\*b\*, quanto cada pixel se afasta do verde da própria folha, rumo a amarelo, laranja ou
   marrom.
4. **Hough de círculos** sobre o mapa: até 60 candidatos a lesão, com raio de 4 a 48 px.
5. **Atributos:** 40 por círculo (cor, contraste, forma e textura, com LBP e GLCM) e 27 pela folha inteira.
6. **Nível 1, por círculo:** um classificador diz quanto cada círculo se parece com cada classe. Cada círculo herda a
   classe da sua folha.
7. **Nível 2, por folha:** o resumo das pontuações dos círculos, com os atributos da folha, dá a probabilidade de cada
   classe.

| Passo | Modelo padrão (BRACOL) | `--modelo rocole` |
|---|---|---|
| 1. Saudável ou doença | `models/bracol_hough.joblib`: classes saudável, outra doença e ferrugem | `models/rocole_hough.joblib` e `models/rocole_resnet50.pt`: média do detector com a ResNet-50 (peso 0,5); classes saudável, ácaro-vermelho e ferrugem |
| 2. Quais doenças | `models/bracol_doencas.joblib`: um classificador por doença | as classes de doença do passo 1 |
| Severidade | área com lesão | área com lesão |

O detector do RoCoLe e a fusão com a ResNet-50 estão descritos no [relatório do RoCoLe](RELATORIO_ROCOLE.md).

## Primeiro passo: saudável ou doença

P(doença) = 1 − P(saudável), e vale a maior das duas. Se nenhuma passa de 60%, a resposta é "Incerto", com o palpite.

| | BRACOL | RoCoLe (fusão) |
|---|---|---|
| Acerto no teste | **99,5%** (98,5% a 100%) | **96,6%** (92,3% a 99,1%) |
| Doença achada no teste | 99,4% (98,3% a 100%) | 93,2% (86,1% a 98,1%) |
| Saudável certa no teste | 100% (21 de 21) | 100% (105 de 105) |
| "Incerto" no teste | 0% | 1,4% (3 de 208) |
| Acerto fora da dobra | 99,1% | 95,8% |
| "Incerto" fora da dobra | 0,8% (acerto de 78% nelas) | 3,8% (acerto de 69% nelas) |

- **Detector do BRACOL:** a variante foi escolhida em validação cruzada (3 dobras, regra fixada antes: média maior e
  vitória em pelo menos 2 das 3 dobras). F1 de ferrugem de 0,939 fora da dobra; sem os círculos do Hough, 0,870. No
  teste, F1 de ferrugem de **0,949** (IC 95% 0,912 a 0,979): 83 de 86 folhas com ferrugem achadas, nenhuma saudável e
  6 das 94 com outra doença chamadas de ferrugem.
- **RoCoLe:** a classe da doença sai certa em 83% das fotos com doença achada no teste. O erro é ácaro-vermelho contra
  ferrugem.

Arquivos: `outputs/hough_bracol/` e `outputs/hough/` (`aplicativo_dois_passos.json`, `cv_resumo.csv`, `cv_oof.csv`,
`decisao.json`, `resultados_teste.json`, `predicoes_teste.csv` e os logs).

## Segundo passo: quais doenças

Quatro classificadores binários (árvores extremamente aleatórias), um por doença, cada um com o seu limiar (bicho-mineiro
0,42, ferrugem 0,52, mancha de phoma 0,44, cercosporiose 0,40), escolhido pelo maior F1 fora da dobra. Treinados nas
1.077 folhas com doença do treino + validação.

- **Atributos:** os do detector do BRACOL e um **nível 1 por doença**: um classificador de círculos com 5 classes
  (saudável e as quatro doenças), treinado com folhas de uma doença só. Para cada classe entram as pontuações dos
  círculos e os atributos dos 5 círculos mais parecidos com ela.
- **Validação cruzada** (mesmas 3 dobras e mesma regra): F1 macro de 0,825 fora da dobra, contra 0,810 do mesmo
  modelo sem o nível 1 por doença, com vitória em 2 das 3 dobras. O ganho veio do bicho-mineiro, que melhorou nas 3.

**Teste** (180 folhas com doença; IC 95% por bootstrap de grupos):

| Métrica | Valor | IC 95% |
|---|---|---|
| F1 macro | **0,860** | 0,815 a 0,899 |
| Acerto exato do conjunto de doenças | 78,3% | 72,1% a 84,4% |
| A doença mais provável está entre as da folha | 96,7% | |

| Doença | Positivos | Precisão | Recall | F1 |
|---|---|---|---|---|
| Bicho-mineiro | 49 | 0,900 | 0,735 | 0,809 |
| Ferrugem | 86 | 0,988 | 0,930 | 0,958 |
| Mancha de phoma | 51 | 0,942 | 0,961 | 0,951 |
| Cercosporiose | 44 | 0,643 | 0,818 | 0,720 |

- A cercosporiose aparece quase sempre com ferrugem (164 folhas com as duas no treino + validação, 66 só com ela), e
  as manchas das duas se confundem. São só 44 folhas com cercosporiose no teste.
- A mina do bicho-mineiro é alongada, e o Hough procura círculos.
- O modelo sem o nível 1 por doença também foi testado uma vez antes da troca: F1 macro 0,834. A troca foi decidida
  pela validação cruzada.

Arquivos em `outputs/hough_bracol/`: `doencas_decisao.json`, `doencas_cv_oof.csv`, `doencas_cv_resumo.csv`,
`doencas_cv_por_doenca.json`, `doencas_cv_por_doenca_oof.csv`, `doencas_cv_por_doenca_resumo.csv`,
`doencas_teste.json`, `doencas_teste_por_doenca.csv`, `doencas_predicoes_teste.csv` e os logs `log_doencas_*.txt`.

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
python -P app/app.py                   # modelos do BRACOL (padrao); depois abra http://127.0.0.1:5000
python -P app/app.py --modelo rocole   # modelos do RoCoLe (fotos de campo)
```

- Envie até 12 fotos por vez (JPG ou PNG), com a folha no centro. A rotação do celular (EXIF) é corrigida.
- Para cada foto: saudável, doença ou incerto; se for doença, a probabilidade de cada doença, com o limiar dela, e a
  severidade estimada. Uma folha saudável mostra só o primeiro passo.
- A foto volta com o contorno da folha e os círculos do Hough, cada um na cor da classe com que mais se parece. Modelo
  padrão: azul, folha sadia; amarelo, bicho-mineiro; vermelho, ferrugem; magenta, mancha de phoma; ciano,
  cercosporiose. `--modelo rocole`: azul, folha sadia; amarelo, ácaro-vermelho; vermelho, ferrugem.
- Um aviso aparece quando o SAM não acha a folha ou quando nenhum círculo é encontrado. Com várias fotos, um resumo
  conta as saudáveis, as com doença (e quantas de cada doença) e as incertas.
- As fotos são analisadas uma por vez. O rodapé diz em que fotos o modelo carregado foi treinado. `--porta` troca a
  porta; `--sem-sam` e `--sem-resnet` desligam essas partes.
- Precisa dos arquivos de `models/` e do SAM (cerca de 375 MB, baixado na primeira vez). Em uma RTX 4050, com o modelo
  do RoCoLe, a primeira análise de 4 fotos levou 27 s (carrega os modelos); depois, cerca de 1,5 s por foto. Só atende
  a própria máquina.
- **Conferido com os modelos de verdade** em folhas de teste: a 104 do BRACOL (ferrugem) saiu com ferrugem em 86% e
  severidade muito baixa, como no rótulo; a 20 (mancha de phoma) saiu com phoma em 78% e severidade alta, um nível
  acima do rótulo; a 1300 (ferrugem e cercosporiose) saiu com as duas doenças e severidade baixa, como no rótulo. No
  RoCoLe, a C12P23H2 (ferrugem) saiu "Doença", com ferrugem, e a C2P24H1 (saudável) saiu "Incerto", com palpite
  "Saudável" (56%). Telas em `docs/app/`.

## Limitações

- **Cada modelo vale para o tipo de foto em que foi treinado.** Não há um número que valha para qualquer folha de café.
  Arábica fotografada na planta e folhas de outras regiões não foram medidas.
- **Testes pequenos:** 201 folhas no BRACOL e 208 fotos no RoCoLe. Os intervalos são largos.
- **Rótulo por folha, não por lesão:** cada círculo herda a classe da sua folha no treino.
- **Bicho-mineiro e cercosporiose** são as doenças mais difíceis (ver acima).
- **Severidade:** conta a área de todas as lesões juntas, sem separar por doença. Foi conferida só na validação do
  BRACOL.
- **O Hough procura círculos.** Lesões grandes, alongadas ou na borda da folha são cobertas por vários círculos.
- **Contorno da folha:** se a folha não está no centro da foto, ou se há várias folhas, o contorno sai errado.
- **`--modelo rocole`** não tem o modelo das doenças: mostra só ácaro-vermelho e ferrugem, que se confundem.
- Roda na própria máquina, uma foto por vez, e foi testado só com GPU.

## Trabalhos futuros

- Fotos do uso real: arábica fotografada na planta, de outras lavouras e regiões.
- Um modelo único com vários conjuntos, medido deixando um conjunto inteiro de fora a cada vez.
- Bicho-mineiro: atributos de forma das manchas do mapa de lesão (alongamento, área, contorno).
- Cercosporiose: medir o centro claro e o anel escuro em volta de cada círculo; reunir mais folhas só com ela.
- Rótulo por lesão em algumas centenas de folhas, para treinar o nível 1 de verdade e dar a severidade por doença.
- Versão para celular, sem GPU, com um segmentador mais leve que o SAM.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `app/app.py`, `app/templates/` | aplicativo local (Flask) |
| `src/hough_inferir.py` | aplica os modelos a uma foto solta (SAM na hora) |
| `src/hough.py` | da foto aos círculos do Hough e aos atributos |
| `src/hough_treinar.py` | extração em lote, modelo de dois níveis, validação cruzada, fusão com a ResNet-50, modelo final |
| `src/hough_doencas.py` | modelo das doenças: validação cruzada, comparação, adoção, modelo final e teste |
| `src/hough_avaliar.py` | teste de cada detector, uma vez |
| `src/hough_dados.py` | os conjuntos (`--dataset rocole` ou `bracol`): fotos, classes, divisão, pastas |
| `src/hough_recusa.py` | faixa de recusa do detector do RoCoLe |
| `src/folha_sam.py` | segmentação da folha com o SAM |
| `src/comum.py` | caminhos e constantes |
| `src/cnn_*.py`, `src/gradcam.py` | ResNet-50: modelo, treino, avaliação, validação cruzada e Grad-CAM |
| `src/preparar_rocole.py`, `src/regras.py` | preparação do RoCoLe e detector por regras (comparação) |
| `tests/` | testes (`unittest`) |
| `models/` | modelos do aplicativo |
| `outputs/hough_bracol/`, `outputs/hough/` | resultados dos modelos do BRACOL e do RoCoLe |
| `outputs/resnet50/`, `outputs/cv_entradas/`, `outputs/regras/` | ResNet-50, validação cruzada da fusão e detector por regras |
| `docs/` | relatórios, resultados, manifesto, explicação em Word, telas e figuras |

O aplicativo usa `hough_inferir.py`, `hough.py`, `hough_treinar.py`, `hough_dados.py`, `hough_doencas.py`,
`folha_sam.py`, `comum.py` e, com `--modelo rocole`, `cnn_modelo.py`. Os demais módulos servem para treinar e avaliar.

## Como reproduzir

Requer Python 3.13 e, de preferência, GPU NVIDIA. Baixe o BRACOL (<https://data.mendeley.com/datasets/yy2k5y8mxg>) e
extraia o arquivo interno em `data/bracol_raw/`. O zip está truncado na própria fonte; `tar -xf arquivo.zip` extrai as
1.401 imagens que existem. Rode da raiz do repositório (no PowerShell: `$env:PYTHONPATH = "src"`):

```bash
export PYTHONPATH=src
python -P src/hough_treinar.py mascaras --bracol   # SAM nas 1.401 folhas (GPU, ~15 min)
python -P src/hough_treinar.py extrair --bracol    # circulos do Hough e atributos
python -P src/hough_treinar.py cv --bracol         # validacao cruzada e escolha da variante
python -P src/hough_treinar.py final --bracol      # grava models/bracol_hough.joblib
python -P src/hough_avaliar.py --bracol            # teste do detector, uma vez
python -P src/hough_doencas.py cv                  # modelo das doencas: base, 3 classificadores x 3 dobras
python -P src/hough_doencas.py comparar            # nivel 1 por doenca contra a base, nas mesmas dobras
python -P src/hough_doencas.py adotar              # adota se passar a regra e escolhe os limiares
python -P src/hough_doencas.py final               # grava models/bracol_doencas.joblib
python -P src/hough_doencas.py teste               # teste do modelo das doencas, uma vez
python -P app/app.py                               # aplicativo em http://127.0.0.1:5000
python -P -m unittest discover -s tests            # testes
```

Treinar de novo na GPU dá números próximos, mas não idênticos. Os modelos do RoCoLe se reproduzem pelo
[relatório do RoCoLe](RELATORIO_ROCOLE.md#como-reproduzir).

## Referências

- Krohling, R. A., Esgario, J. e Ventura, J. A. *BRACOL: A Brazilian Arabica Coffee Leaf images dataset*. Mendeley
  Data, 2019.
- Esgario, J. G. M., Krohling, R. A. e Ventura, J. A. *Deep learning for classification and severity estimation of
  coffee leaf biotic stress*. Computers and Electronics in Agriculture 169, 105162, 2020 (faixas de severidade).
- Parraga-Alava, J. et al. *RoCoLe: A robusta coffee leaf images dataset for evaluation of machine learning based
  methods in plant diseases recognition*. Data in Brief 25, 104414, 2019.
- Kirillov, A. et al. *Segment Anything*. ICCV, 2023 (modelo `facebook/sam-vit-base`).
- He, K. et al. *Deep Residual Learning for Image Recognition*. CVPR, 2016 (ResNet-50).
- Duda, R. O. e Hart, P. E. *Use of the Hough Transformation to Detect Lines and Curves in Pictures*.
  Communications of the ACM 15(1), 1972.
- Yuen, H. K. et al. *Comparative study of Hough Transform methods for circle finding*. Image and Vision Computing
  8(1), 1990.
- Ojala, T., Pietikäinen, M. e Mäenpää, T. *Multiresolution Gray-Scale and Rotation Invariant Texture Classification
  with Local Binary Patterns*. IEEE TPAMI 24(7), 2002 (LBP).
- Haralick, R. M., Shanmugam, K. e Dinstein, I. *Textural Features for Image Classification*. IEEE Transactions on
  Systems, Man, and Cybernetics 3(6), 1973 (GLCM).
- Geurts, P., Ernst, D. e Wehenkel, L. *Extremely randomized trees*. Machine Learning 63, 2006.
