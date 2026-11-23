# Resultados

Cada número foi lido do arquivo indicado. O teste foi usado uma vez por modelo; modelo, limiares e regras foram
escolhidos na validação ou em validação cruzada no treino + validação.

## 1. Aplicativo

### Saudável ou doença

P(doença) = 1 − P(saudável); vale a maior das duas. "Incerto" quando nenhuma passa de 60%.
Arquivos: `outputs/hough_bracol/aplicativo_dois_passos.json` e `outputs/hough/aplicativo_dois_passos.json`.

| | BRACOL (modelo padrão) | RoCoLe (`--modelo rocole`) |
|---|---|---|
| Acerto no teste | **99,5%** (98,5% a 100%) | **96,6%** (92,3% a 99,1%) |
| Doença achada no teste | 99,4% (98,3% a 100%) | 93,2% (86,1% a 98,1%) |
| Saudável certa no teste | 100% (21 de 21) | 100% (105 de 105) |
| "Incerto" no teste | 0% | 1,4% (3 de 208) |
| Acerto fora da dobra | 99,1% | 95,8% |
| "Incerto" fora da dobra | 0,8% (acerto de 78% nelas) | 3,8% (acerto de 69% nelas) |

### Quais doenças (BRACOL)

Teste com as 180 folhas que têm doença.
Arquivos: `outputs/hough_bracol/doencas_teste.json` e `doencas_teste_por_doenca.csv`.

| Métrica | Valor | IC 95% |
|---|---|---|
| F1 macro | **0,860** | 0,815 a 0,899 |
| Acerto exato do conjunto de doenças | 78,3% | 72,1% a 84,4% |
| A doença mais provável está entre as da folha | 96,7% | |
| F1 macro fora da dobra | 0,825 | |

| Doença | Positivos | Precisão | Recall | F1 | AP |
|---|---|---|---|---|---|
| Bicho-mineiro | 49 | 0,900 | 0,735 | 0,809 | 0,917 |
| Ferrugem | 86 | 0,988 | 0,930 | 0,958 | 0,991 |
| Mancha de phoma | 51 | 0,942 | 0,961 | 0,951 | 0,990 |
| Cercosporiose | 44 | 0,643 | 0,818 | 0,720 | 0,824 |

O mesmo modelo sem o nível 1 por doença teve F1 macro 0,834 no teste e 0,810 fora da dobra. A troca foi decidida pela
validação cruzada (`doencas_cv_por_doenca_resumo.csv`).

### Severidade (BRACOL)

Fração da folha com cor de lesão, nas faixas de área do BRACOL; nenhum corte ajustado aos dados. Validação com 200
folhas; o teste não foi usado. Arquivo: `outputs/hough_bracol/severidade_validacao.json`.

| Métrica | Valor |
|---|---|
| Nível certo | 83,5% |
| Até um nível de distância | 98,0% |
| Kappa quadrático | 0,785 |
| Acerto de quem chuta sempre a classe mais comum | 61,5% |

## 2. Detectores do primeiro passo

### BRACOL: detector com o Hough

Teste com 201 folhas, 86 com ferrugem.
Arquivos: `outputs/hough_bracol/resultados_teste.json` e `teste_intervalos_confianca.csv`.

| Métrica | Valor | IC 95% |
|---|---|---|
| F1 de ferrugem | 0,949 | 0,912 a 0,979 |
| Sensibilidade | 0,965 (83 de 86) | 0,921 a 1,000 |
| Especificidade | 0,948 (6 alarmes em 115) | 0,906 a 0,983 |
| F1 fora da dobra | 0,939 | |

### RoCoLe: detector com o Hough e ResNet-50

Média das probabilidades do detector e da ResNet-50. Teste com 208 fotos, 81 com ferrugem.
Arquivos: `outputs/hough/resultados_teste_fusao.json` e `teste_intervalos_confianca_fusao.csv`.

| Métrica | Valor | IC 95% |
|---|---|---|
| F1 de ferrugem | 0,879 | 0,774 a 0,934 |
| Sensibilidade | 0,938 | 0,855 a 0,987 |
| Especificidade | 0,874 | 0,784 a 0,935 |
| F1 fora da dobra | 0,847 | |

## 3. Outros modelos e comparações

### RoCoLe: comparação e recusa

Arquivos: `outputs/hough/resultados_teste.json`, `outputs/resnet50/resultados_teste.json`,
`outputs/regras/resultados_teste.json`, `outputs/hough/recusa_teste.json` e `recusa_teste_ic.csv`.

| F1 de ferrugem no teste | Fusão | Detector com o Hough | ResNet-50 | Detector por regras |
|---|---|---|---|---|
| F1 | **0,879** | 0,876 (0,761 a 0,941) | 0,835 (0,700 a 0,906) | 0,476 (0,322 a 0,559) |

Com recusa nas fotos em dúvida, o detector responde a 86,1% das fotos (IC 95% 75,6% a 92,6%), com F1 de **0,935**
(IC 95% 0,845 a 0,980) nas aceitas.

### RoCoLe: ResNet-50 em três classes

ResNet-50 com entrada 512×288. Teste com 208 fotos.
Arquivos: `outputs/resnet50/resultados_teste.json` e `teste_intervalos_confianca.csv`.

| Métrica | Valor | IC 95% |
|---|---|---|
| F1 macro | 0,753 | 0,650 a 0,829 |
| Acurácia | 86,1% | 77,0% a 91,7% |
| F1 Saudável | 0,958 | 0,915 a 0,982 |
| F1 Ácaro-vermelho | 0,465 | 0,216 a 0,654 |
| F1 Ferrugem | 0,835 | 0,700 a 0,906 |

### BRACOL: quatro doenças e severidade (ConvNeXt-Tiny)

Experimento anterior ao aplicativo. Entrada 512×256; teste com 201 folhas. Os arquivos ficam no projeto original
(`outputs/resultados_teste.json`, `outputs/teste_intervalos_confianca.csv` e `outputs/teste_por_doenca.csv` de lá).

| Métrica | Valor | IC 95% |
|---|---|---|
| Acerto exato do conjunto de doenças | 84,6% | 79,6% a 89,6% |
| mAP | 0,950 | 0,917 a 0,979 |
| F1 macro | 0,904 | 0,871 a 0,935 |
| Severidade (acurácia) | 83,1% | 78,1% a 88,1% |
| Acerto nas folhas aceitas (77,6% aceitas) | 92,3% | |
