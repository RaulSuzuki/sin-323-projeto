# Modelos do aplicativo

| Arquivo | Conteúdo |
|---|---|
| `bracol_hough.joblib` | passo 1, saudável ou doença: modelo de Hough de dois níveis treinado no BRACOL |
| `bracol_doencas.joblib` | passo 2, quais doenças: um classificador por doença, o nível 1 por doença, o peso da média com a ConvNeXt-Tiny e os limiares |
| `bracol_convnext.pt` | pesos da ConvNeXt-Tiny (4 doenças), treinada no treino + validação do BRACOL |
| `bracol_convnext.json` | impressão digital da divisão de dados e melhor época do treino |

## Experimento anterior (não usados pelo aplicativo)

| Arquivo | Conteúdo |
|---|---|
| `rocole_hough.joblib` | modelo de Hough do RoCoLe, com peso da fusão, limiar e faixa de recusa |
| `rocole_resnet50.pt` | pesos da ResNet-50 treinada no RoCoLe, usada na fusão |
| `rocole_resnet50.json` | impressão digital da divisão de dados usada no treino |
