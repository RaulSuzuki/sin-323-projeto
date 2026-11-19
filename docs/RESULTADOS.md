# Resultados finais

Todos os números abaixo foram lidos dos arquivos indicados, nesta mesma pasta. O teste foi usado uma vez por modelo
final. As escolhas de modelo, limiar e recusa vieram da validação ou da validação cruzada.

## 1. Quatro doenças e severidade (BRACOL)

ConvNeXt-Tiny com entrada 512×256. Teste com 201 folhas. O código deste modelo está no projeto original.
Arquivos: `resultados_4_doencas/outputs/resultados_teste.json`, `resultados_4_doencas/outputs/teste_intervalos_confianca.csv`, `resultados_4_doencas/outputs/teste_por_doenca.csv`.

| Métrica | Valor | IC 95% |
|---|---|---|
| Acerto exato do conjunto de doenças | 84,6% | 79,6% a 89,6% |
| mAP | 0,950 | 0,917 a 0,979 |
| F1 macro | 0,904 | 0,871 a 0,935 |
| Severidade (acurácia) | 83,1% | 78,1% a 88,1% |
| Acerto nas folhas aceitas (77,6% aceitas) | 92,3% | |

| Doença | Positivos | Precisão | Recall | F1 | AP |
|---|---|---|---|---|---|
| Bicho-mineiro | 49 | 0,827 | 0,878 | 0,851 | 0,934 |
| Ferrugem | 86 | 0,988 | 0,988 | 0,988 | 1,000 |
| Mancha de Phoma | 51 | 0,979 | 0,922 | 0,949 | 0,990 |
| Cercosporiose | 44 | 0,745 | 0,932 | 0,828 | 0,875 |

Figuras: [Grad-CAM](../resultados_4_doencas/outputs/gradcam_exemplos.png), [erros](../resultados_4_doencas/outputs/erros_gradcam.png) e
[robustez](../resultados_4_doencas/outputs/teste_robustez.png).

## 2. Detector de ferrugem no BRACOL

Modelo de Hough sozinho. Teste com 201 folhas, 86 com ferrugem.
Arquivos: `outputs/hough_bracol/resultados_teste.json` e `teste_intervalos_confianca.csv`.

| Métrica | Valor | IC 95% |
|---|---|---|
| F1 de ferrugem | 0,949 | 0,912 a 0,979 |
| Sensibilidade | 0,965 (83 de 86) | 0,921 a 1,000 |
| Especificidade | 0,948 (6 alarmes em 115) | 0,906 a 0,983 |
| F1 fora da dobra (treino + validação) | 0,939 | |

## 3. Detector de ferrugem no RoCoLe

Fusão do modelo de Hough com a ResNet-50. Teste com 208 fotos, 81 com ferrugem.
Arquivos: `outputs/hough/resultados_teste_fusao.json`, `teste_intervalos_confianca_fusao.csv`,
`recusa_teste.json`, `recusa_teste_ic.csv`.

| Ferrugem, foto a foto | Fusão | Modelo de Hough | ResNet-50 | Detector por regras |
|---|---|---|---|---|
| **F1** | **0,879** (0,774 a 0,934) | 0,876 (0,761 a 0,941) | 0,835 (0,700 a 0,906) | 0,476 (0,322 a 0,559) |
| Sensibilidade | 0,938 | 0,963 | 0,815 | 0,432 |
| Especificidade | 0,874 | 0,850 | 0,913 | 0,756 |

Com a **opção de recusa**, o detector responde a 86,1% das fotos (IC 95% 75,6% a 92,6%) e o F1 nas fotos
aceitas é **0,935** (IC 95% 0,845 a 0,980). Fora da dobra, o F1 é 0,847 sem recusa e 0,911 com
recusa (79,3% das fotos aceitas).

## 4. Três classes em fotos de campo (RoCoLe)

ResNet-50 com entrada 512×288. Teste com 208 fotos.
Arquivos: `outputs/resnet50/resultados_teste.json` e `teste_intervalos_confianca.csv`.

| Métrica | Valor | IC 95% |
|---|---|---|
| F1 macro | 0,753 | 0,650 a 0,829 |
| Acurácia | 86,1% | 77,0% a 91,7% |
| F1 Saudável | 0,958 | 0,915 a 0,982 |
| F1 Ácaro-vermelho | 0,465 | 0,216 a 0,654 |
| F1 Ferrugem | 0,835 | 0,700 a 0,906 |
