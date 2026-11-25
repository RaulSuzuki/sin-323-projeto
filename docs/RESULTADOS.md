# Resultados

Cada número foi lido do arquivo indicado, todos em `outputs/hough_bracol/`. Modelo, limiares e pesos foram
escolhidos em validação cruzada no treino + validação do BRACOL; o teste foi lido uma vez, no fim.

## Passo 1: saudável ou doença

P(doença) = 1 − P(saudável); vale a maior das duas. "Incerto" quando nenhuma passa de 60%.
Arquivos: `resultados_teste.json` e `teste_intervalos_confianca.csv`.

| | Teste (201 folhas) | IC 95% | Fora da dobra (1.200 folhas) |
|---|---|---|---|
| Acerto | **99,0%** | 97,5% a 100% | 98,9% |
| Doenças achadas | 98,9% (178 de 180) | 97,2% a 100% | 99,0% (1.066 de 1.077) |
| Saudáveis certas | 100% (21 de 21) | | 98,4% (121 de 123) |
| "Incerto" | 0% | | 0,1% (1 folha) |

As 2 folhas com doença que viraram "saudável" no teste (104 e 1079) têm só ferrugem, com severidade muito baixa.

Validação cruzada (F1 macro de saudável × doença; `cv_resumo.csv`, `decisao.json`): a base `completo` ficou com
**0,972**; nenhuma alternativa passou a regra. Sem os círculos do Hough, 0,969; só com os círculos, 0,951.

## Passo 2: quais doenças

Teste com as 180 folhas que têm doença. Árvores (um classificador por doença, com o nível 1 por doença) em média com a
ConvNeXt-Tiny, peso 0,25 nas árvores.
Arquivos: `doencas_teste.json`, `doencas_teste_por_doenca.csv` e `doencas_predicoes_teste.csv`.

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

### Como o modelo foi escolhido

F1 macro fora da dobra, nas 1.077 folhas com doença do treino + validação, 3 dobras. Uma opção só substitui a atual
com média maior e vitória em pelo menos 2 das 3 dobras.

| Etapa | Opção | Média | Arquivo |
|---|---|---|---|
| Classificador | gradient boosting (adotado) / árvores extremamente aleatórias / regressão logística | 0,745 / 0,728 / 0,731 | `doencas_cv_resumo.csv` |
| Nível 1 por doença | sem / com (adotado) | 0,745 / 0,825 | `doencas_cv_por_doenca_resumo.csv` |
| Média com a ConvNeXt-Tiny | só árvores / peso 0,25 (adotado) / 0,5 / 0,75 | 0,825 / 0,876 / 0,870 / 0,848 | `doencas_fusao_resumo.csv` |
| Referência | ConvNeXt-Tiny sozinha | 0,858 | `doencas_fusao_resumo.csv` |

- ConvNeXt-Tiny fora da dobra: mAP 0,929 (`convnext_oof.csv`). Rede final: melhor época 18 (`models/bracol_convnext.json`).
- Limiares da média: bicho-mineiro 0,66, ferrugem 0,40, mancha de phoma 0,54 e cercosporiose 0,48
  (`doencas_decisao.json`).
- Só para descrever (nada foi escolhido com isso): no mesmo teste, as árvores sozinhas tiveram F1 macro de 0,861.

## Severidade

Fração da folha com cor de lesão, nas faixas de área do BRACOL; nenhum corte ajustado aos dados. Validação com 200
folhas; o teste não foi usado. Arquivo: `severidade_validacao.json`.

| Métrica | Valor |
|---|---|
| Nível certo | 83,5% |
| Até um nível de distância | 98,0% |
| Kappa quadrático | 0,785 |
| Acerto de quem chuta sempre a classe mais comum | 61,5% |

Os resultados dos experimentos anteriores estão em [RELATORIO_BRACOL.md](RELATORIO_BRACOL.md) e
[RELATORIO_ROCOLE.md](RELATORIO_ROCOLE.md).
