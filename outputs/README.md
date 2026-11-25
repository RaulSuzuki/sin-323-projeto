# Resultados

## BRACOL: o aplicativo

Tudo em `hough_bracol/`.

| Arquivo | Conteúdo |
|---|---|
| `metadados.csv` | divisão do BRACOL em treino, validação e teste (agrupada por quase-cópias) |
| `folhas_sam.csv` | contorno da folha pelo SAM: confiança de cada folha |
| `cv_oof.csv` | passo 1 (saudável ou doença): previsões fora da dobra do modelo |
| `resultados_teste.json`, `predicoes_teste.csv`, `teste_intervalos_confianca.csv` | passo 1 no teste |
| `doencas_cv_oof.csv`, `doencas_decisao.json` | passo 2 (quais doenças): previsões fora da dobra e limiar de cada doença |
| `convnext_oof.csv`, `convnext_historico_*.csv` | ConvNeXt-Tiny: previsões fora da dobra e histórico do treino |
| `doencas_teste.json`, `doencas_teste_por_doenca.csv`, `doencas_predicoes_teste.csv` | passo 2 no teste |
| `severidade_validacao.json` | severidade pela área com lesão, conferida na validação |
| `propostas_treino.jpg` | exemplos de círculos do Hough em folhas de treino |
| `log_*.txt` | saída de cada comando |

Registro de como a configuração do aplicativo foi escolhida (o código de hoje treina só a escolhida):

| Arquivo | Conteúdo |
|---|---|
| `cv_resumo.csv`, `decisao.json` | passo 1: as variantes comparadas na validação cruzada e a escolhida |
| `doencas_cv_resumo.csv` | passo 2: gradient boosting contra árvores extremamente aleatórias e regressão logística |
| `doencas_cv_por_doenca_*` | passo 2: com e sem o nível 1 por doença |
| `doencas_fusao_oof.csv`, `doencas_fusao_resumo.csv` | passo 2: árvores contra a média com a ConvNeXt-Tiny, com pesos 0,25, 0,5 e 0,75 |
| `log_doencas_comparar.txt`, `log_doencas_adotar.txt`, `log_doencas_fusao.txt` | saída desses comandos |

## Experimento anterior: RoCoLe

Não usado pelo aplicativo. Descrito em `docs/RELATORIO_ROCOLE.md`.

| Pasta ou arquivo | Conteúdo |
|---|---|
| `hough/` | detector de ferrugem do RoCoLe: validação cruzada, fusão, recusa, teste e conferência no BRACOL |
| `resnet50/` | ResNet-50 de 3 classes: métricas, matriz de confusão, Grad-CAM e diagnóstico pela folha inteira |
| `cv_entradas/` | decisão entre foto inteira e recortes, com as previsões usadas na fusão |
| `regras/` | detector por regras, usado como comparação |
| `metadados.csv`, `descartadas.csv` | divisão do RoCoLe e imagens descartadas |
| `folhas_sam.csv`, `segmentacao_amostra.jpg`, `log_segmentacao.txt` | segmentação da folha com o SAM no RoCoLe |
| `comparacao_modelos.csv` | validação e teste por modelo |
