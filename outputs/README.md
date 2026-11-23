# Resultados do detector e do aplicativo

| Pasta ou arquivo | Conteúdo |
|---|---|
| `hough/` | detector do RoCoLe: validação cruzada, fusão, recusa, teste, conferência no BRACOL e a resposta em dois passos do aplicativo (`aplicativo_dois_passos.json`) |
| `hough_bracol/` | detector treinado e testado no BRACOL, com a divisão desse conjunto (`metadados.csv`); modelo das doenças (`doencas_*`), resposta em dois passos (`aplicativo_dois_passos.json`) e severidade estimada na validação (`severidade_validacao.json`) |
| `resnet50/` | ResNet-50 de 3 classes: métricas, matriz de confusão, Grad-CAM e diagnóstico pela folha inteira |
| `cv_entradas/` | decisão entre foto inteira e recortes, com as previsões usadas na fusão |
| `regras/` | detector por regras, usado como comparação |
| `metadados.csv`, `descartadas.csv` | divisão do RoCoLe e imagens descartadas |
| `folhas_sam.csv`, `segmentacao_amostra.jpg`, `log_segmentacao.txt` | segmentação da folha com o SAM |
| `comparacao_modelos.csv` | validação e teste por modelo |
