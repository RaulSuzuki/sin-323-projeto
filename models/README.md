# Modelos finais do aplicativo

| Arquivo | Conteúdo |
|---|---|
| `bracol_hough.joblib` | modelo de Hough treinado no BRACOL (padrão do aplicativo: saudável ou doença) |
| `bracol_doencas.joblib` | modelo das doenças do BRACOL: um classificador por doença e o nível 1 por doença (padrão do aplicativo: quais doenças) |
| `rocole_hough.joblib` | modelo de Hough do RoCoLe, com peso da fusão, limiar e faixa de recusa |
| `rocole_resnet50.pt` | pesos da ResNet-50 treinada no RoCoLe, usada na fusão |
| `rocole_resnet50.json` | impressão digital da divisão de dados usada no treino |
