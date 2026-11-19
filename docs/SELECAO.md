# O que entrou nesta pasta, e por quê

Esta pasta é a versão final do projeto, dedicada ao **detector de ferrugem e ao aplicativo**. O projeto original continua
intacto, com todos os experimentos. O critério de "melhor resultado" é o do próprio projeto: **um modelo só é adotado se
vencer na validação, seguindo uma regra fixada antes**. Maior número no teste não basta, porque escolher pelo teste
invalidaria a medida.

## O que entrou

| Item | Motivo |
|---|---|
| Modelo de Hough do BRACOL e resultados | detector de ferrugem de maior F1 e modelo padrão do aplicativo |
| Fusão Hough + ResNet-50 do RoCoLe, com recusa, e resultados | detector final para fotos de campo |
| ResNet-50 do RoCoLe e resultados | modelo final de 3 classes e metade da fusão |
| `app/`, `src/`, `tests/` | aplicativo, código do detector e testes |
| `outputs/hough_bracol/metadados.csv` | divisão do BRACOL, lida pelo aplicativo e pelo treino |
| `outputs/cv_entradas/`: decisão, resultados e previsões fora da dobra | a etapa da fusão lê estas previsões |
| `outputs/regras/`: previsões, resultados e intervalos | a avaliação final usa o detector por regras como comparação |
| `docs/specs/` | regras e métodos definidos antes de rodar os experimentos |
| `resultados_4_doencas/` | pesos e resultados do modelo de 4 doenças e severidade |

## O que ficou de fora (continua no projeto original)

| Item | Tamanho | Motivo |
|---|---|---|
| `src/` da raiz original | 60 KB | código do modelo de 4 doenças, que o aplicativo não usa |
| `old/` | 3,7 GB | experimentos que não foram adotados |
| `data/` | 862 MB | dados públicos e caches gerados; baixe das fontes |
| `rocole/models/cv/` | 849 MB | 9 modelos usados só na validação cruzada |
| `rocole/outputs/regras_folha/` | 80 KB | variante do detector por regras que não foi ao teste |
| Demais arquivos de `rocole/outputs/regras/` e `cv_entradas/` | 2,7 MB | ajustes e históricos de abordagens não adotadas |
| `rocole/docs/plans/` | | listas de tarefas de implementação |

Para repetir a etapa de fusão do zero, é preciso recriar `models/cv/` com `src/rocole_cv.py` ou copiá-la do projeto
original. Os resultados e o modelo final da fusão já estão aqui.

## Mudanças em relação ao projeto original

- O conteúdo de `rocole/` passou para a raiz. Para isso, os caminhos foram ajustados em `src/rocole_comum.py` e
  `src/rocole_hough_dados.py`, e os comentários de uso que citavam `rocole/` foram atualizados. A lógica não mudou.
- `app/templates/index.html` tem um visual novo.

## Como conferir

`MANIFESTO.csv` lista cada arquivo com sua origem, tamanho e SHA-256. A coluna `situacao` indica `copia identica`,
`adaptado` ou `novo`.
