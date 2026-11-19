# RoCoLe: doenças do café em fotos de campo

Parte do projeto de SIN 323 (diagnóstico de doenças do café por foto de folha; o modelo de 4 doenças, com o BRACOL,
tem pesos e resultados em `resultados_4_doencas/`). Aqui o modelo é separado (outras classes, outra espécie) e serve para medir o
desempenho em fotos parecidas com as do aplicativo: folhas fotografadas **na planta**, com fundo de campo.

O foco do detector e do aplicativo desta pasta é a **ferrugem**, em qualquer folha de café: de qualquer região,
espécie e tipo de foto. O que já foi medido em cada tipo de folha está em
[Em que folhas o detector foi medido](#em-que-folhas-o-detector-foi-medido).

> É um **apoio à decisão**. Não substitui o agrônomo.

## Resumo

- **Modelo final: ResNet-50, foto inteira em 512×288.** No teste (208 fotos de plantas que o modelo nunca viu):
  F1 macro **0,753** (IC 95% 0,650 a 0,829), acurácia 86,1%. A folha saudável quase sempre sai certa; o ponto
  fraco é separar ácaro-vermelho de ferrugem (F1 do ácaro 0,465).
- **Diagnóstico pela folha inteira** (foto da face + foto do verso; doença presente se aparece em qualquer lado),
  escolhido na validação e pontuado uma vez no teste: ferrugem achada em 39 das 42 folhas (92,9%) e ácaro em 9 das
  15 (60,0%), contra 37 e 2 só com a foto da face.
- **Detector de ferrugem com a transformada de Hough** (seção própria abaixo): os círculos do Hough viram
  *candidatos a lesão* dentro da folha, descritos por cor, contraste, forma e textura e pontuados por um classificador;
  a resposta final é a média com a ResNet-50. Para a pergunta "há ferrugem na foto?", no teste: F1 **0,879**
  (IC 95% 0,774 a 0,934), contra 0,835 da ResNet-50 sozinha e 0,476 do detector por regras. Fora da dobra
  (treino + validação) fica em 0,847. **A meta de 0,90 não foi atingida foto a foto**; pela folha inteira, com as duas
  fotos, o teste dá 0,901.
- **Opção de recusa** ("mostre a um agrônomo"): quando o detector deixa de responder nas fotos em dúvida (20% delas
  fora da dobra, 14% no teste), o F1 de ferrugem nas fotos aceitas é 0,911 fora da dobra e **0,935** no teste
  (IC 95% 0,845 a 0,980). É assim que a meta de 0,90 é atingida.
- **Um modelo por conjunto** (`--dataset rocole|bracol`): o mesmo código treina e testa em cada conjunto, sem juntar
  os dois. Treinado e testado no RoCoLe, o detector dá os números acima. Treinado e testado no BRACOL (arábica, folha
  destacada), o modelo de Hough sozinho dá F1 **0,949** (IC 95% 0,912 a 0,979), sem fusão nem recusa, e separa a
  ferrugem das outras doenças.
- **Aplicativo local** (`app/app.py`): recebe fotos sem máscara, acha a folha com o SAM e mostra o veredito com os
  círculos. Usa por padrão o modelo do BRACOL; `--modelo rocole` troca para a fusão do RoCoLe, com a resposta
  "Incerto".
- **Cada modelo vale para o tipo de foto em que foi treinado.** Na conferência externa, o detector do RoCoLe achou a
  ferrugem nas 304 folhas do BRACOL que só têm ferrugem, mas também chamou de ferrugem 13,9% das saudáveis e 88% das
  que têm outra doença. Ver [Limitações](#limitações).
- **Testado e não adotado:**
  - deslocar as probabilidades por classe (ganho pequeno e instável na validação);
  - recortar a folha-alvo com o SAM (na validação cruzada, nenhum recorte venceu a foto inteira em 2 das 3 dobras);
  - o detector de ferrugem por regras de Rodriguez-Gallo et al. (2023): acurácia balanceada de 0,594 no teste,
    contra 0,864 da ResNet-50.
- **Cuidados contra vazamento:** a divisão agrupa planta, cópias e refotos da mesma folha. Toda escolha foi feita
  na validação ou em validação cruzada no treino + validação. O teste foi pontuado uma vez por modelo final.

O que foi feito, em ordem:

| Etapa | O que | Resultado |
|---|---|---|
| 1 | Limpeza e divisão agrupada (`src/rocole_preparar.py`) | 1.479 imagens; 1.061 / 210 / 208 |
| 2 | ResNet-50 com Grad-CAM (`src/rocole_treinar.py`, `src/rocole_avaliar.py`) | F1 macro 0,753 no teste |
| 3 | Ajustes na validação (`src/rocole_validacao.py`) | deslocamentos: não; folha inteira: sim |
| 4 | Réplica do detector por regras (`src/rocole_regras.py`) | 0,594 no teste, bem abaixo da CNN |
| 5 | Folha-alvo com o SAM e validação cruzada (`src/rocole_folha.py`, `src/rocole_cv.py`) | fica a foto inteira |
| 6 | Folha inteira no teste (`src/rocole_validacao.py --teste`) | ferrugem 92,9%, ácaro 60,0% |
| 7 | Círculos do Hough e atributos (`src/rocole_hough.py`) | 40 atributos por círculo, 27 por folha |
| 8 | Modelo de dois níveis e validação cruzada (`src/rocole_hough_treinar.py`) | F1 de ferrugem 0,824 fora da dobra (0,752 sem os círculos) |
| 9 | Fusão tardia com a ResNet-50 (`rocole_hough_treinar.py fusao`) | 0,847 fora da dobra; adotada |
| 10 | Teste, uma vez por modelo (`src/rocole_hough_avaliar.py`) | Hough 0,876; fusão 0,879 |
| 11 | Conferência externa no BRACOL e aplicativo (`app/app.py`) | ferrugem 304 de 304; alarme em 13,9% das saudáveis |
| 12 | Validação cruzada repetida, auditoria (`rocole_hough_treinar.py cv-repetida`) | a escolha se mantém; modelo de Hough em 0,807 na média de 5 divisões |
| 13 | Opção de recusa (`src/rocole_hough_recusa.py`) | recusando 20%: F1 0,911 fora da dobra; no teste, 0,935 em 86% das fotos |
| 14 | Treino e teste por conjunto (`--dataset`, `src/rocole_hough_dados.py`) | BRACOL: 0,939 fora da dobra, 0,949 no teste |
| 15 | Aplicativo com o modelo do BRACOL por padrão (`app/app.py --modelo`) | 11 de 12 fotos de conferência certas |

## Dados

RoCoLe (Parraga-Alava et al., *Data in Brief* 25, 104414, 2019; Mendeley Data `c5yvn32dzg`, CC BY 4.0),
na versão do Kaggle (`archive.zip`): 1.560 fotos 1920×1080 de café **robusta**, em 3 classes: saudável, ácaro-vermelho
e ferrugem. É o mesmo conjunto de fotos do artigo original (as contagens batem: 791 saudáveis, 167 com ácaro e 602
com ferrugem). A versão do Kaggle só organiza as fotos em três pastas de classe e não traz as anotações que o
original publica junto: o nível da ferrugem (1 a 4) e o contorno da folha.

- Cada folha tem duas fotos (face `H` e verso `E`) e cada planta (`C<n>P<m>`) tem duas folhas.
- O zip tem 67 pares de arquivos idênticos com nomes diferentes; em 12 os rótulos divergem. Há mais 1 quase-cópia
  (pHash 0) com rótulos diferentes. Usamos **1.479 imagens** (757 saudáveis, 567 com ferrugem, 155 com ácaro); os
  81 descartes estão em `outputs/descartadas.csv`.
- **Divisão 70/15/15 agrupada:** a mesma planta, a mesma cópia e refotos da mesma folha (pHash ≤ 10, calculado em
  todas as imagens antes dos descartes) ficam sempre no mesmo split: 1.061 treino, 210 validação, 208 teste.
  Nenhum par com pHash ≤ 11 atravessa splits; os 4 pares com distância 12 são folhas diferentes (conferido à mão).
  A divisão está em `outputs/metadados.csv`.

## Modelo

ResNet-50 pré-treinado no ImageNet, entrada 512×288 (mantém o 16:9), adaptado do `main.py` do trabalho
de SIN 393: perda com peso por classe, 3 épocas só na última camada e depois a rede inteira (AdamW, AMP), parada
antecipada pela perda de validação. Parou na época 23; a melhor foi a 18. Cerca de 25 min numa RTX 4050.

O checkpoint fica em `models/rocole_resnet50.pt`. Ao lado dele, `models/rocole_resnet50.json` guarda o
SHA-256 do `outputs/metadados.csv` usado no treino; a avaliação para se a divisão tiver mudado.

## Resultados no teste

208 imagens de 37 grupos de plantas que o modelo nunca viu; IC 95% por bootstrap de grupos, porque as duas fotos de
uma folha não são independentes.

| Métrica | Valor | IC 95% |
|---|---|---|
| Acurácia | 86,1% | 77,0% a 91,7% |
| **F1 macro** (métrica principal; classes desbalanceadas) | **0,753** | 0,650 a 0,829 |
| F1 Saudável (105 fotos) | 0,958 | 0,915 a 0,982 |
| F1 Ácaro-vermelho (22 fotos) | 0,465 | 0,216 a 0,654 |
| F1 Ferrugem (81 fotos) | 0,835 | 0,700 a 0,906 |

Validação (210 imagens): acurácia 91,4%, F1 macro 0,864.

| Real \ Predito | Saudável | Ácaro-vermelho | Ferrugem |
|---|---|---|---|
| Saudável | 103 | 0 | 2 |
| Ácaro-vermelho | 3 | 10 | 9 |
| Ferrugem | 4 | 11 | 66 |

- Folha saudável quase sempre certa (103 de 105).
- O ponto fraco é separar ácaro de ferrugem: 20 dos 29 erros são uma no lugar da outra. O ácaro tem só 111 fotos de
  treino e 22 de teste, por isso o intervalo é tão largo.
- A validação ficou acima do teste (F1 macro 0,864 contra 0,753), e a diferença vem quase toda do ácaro. Com tão
  poucas fotos dessa classe, o número de uma única divisão varia muito.
- **Grad-CAM** (`outputs/resnet50/gradcam_acertos.png` e `gradcam_erros.png`): nos acertos, o mapa cai sobre a
  folha e, na ferrugem, sobre as manchas. Nos erros mais confiantes, ácaro e ferrugem disputam as mesmas lesões. O
  mapa tem 9×16 células antes de ser ampliado: mostra a região que pesou, não a lesão individual.

Arquivos: `outputs/resnet50/` (métricas, IC, predições, matriz de confusão, Grad-CAM, histórico e log do
treino) e `outputs/comparacao_modelos.csv` (validação e teste por modelo; a escolha entre modelos, por exemplo
um ConvNeXt-Tiny com `--modelo convnext_tiny`, deve ser feita pela validação).

## Ajustes escolhidos na validação

Os dois ajustes foram escolhidos só na validação (`src/rocole_validacao.py`). Só o que foi adotado, a folha
inteira, foi pontuado no teste, uma vez, com o modelo final (tabela "No teste", abaixo).

- **Deslocar as probabilidades por classe** (deslocamentos do ácaro e da ferrugem escolhidos pelo F1 macro). Medido
  fora da dobra, com 5 dobras por planta dentro da validação, o F1 macro passa de 0,864 para 0,887 e a acurácia de
  91,4% para 93,3% (4 fotos a mais certas, de 210). O recall do ácaro não muda (72,7%), e o deslocamento escolhido
  muda de sinal entre as dobras (ácaro +0,5 numa, −1,0 nas outras). Ganho pequeno e instável: **não adotado**.
- **Diagnosticar pela folha inteira:** foto da face e do verso, doença presente se aparece em qualquer lado. Nas 104
  folhas da validação com as duas fotos:

| Estratégia | Ferrugem: sensibilidade | Ferrugem: especificidade | Ácaro: sensibilidade (15 folhas) | Ácaro: especificidade | Folha certa |
|---|---|---|---|---|---|
| Só a face | 80,9% | 93,0% | 46,7% | 97,8% | 81,7% |
| Só o verso | 76,6% | 98,2% | 66,7% | 97,8% | 85,6% |
| **As duas fotos** | **89,4%** | 93,0% | **80,0%** | 95,5% | **86,5%** |

  "Folha certa" = presença de ferrugem e de ácaro certas ao mesmo tempo. Com as duas fotos, o ácaro é encontrado em
  12 das 15 folhas, contra 7 só com a face, com pouca perda de especificidade. São só 15 folhas com ácaro: era um
  sinal a confirmar no teste com o modelo final.

**No teste** (uma vez, com o modelo final; 99 folhas com as duas fotos, 42 com ferrugem e 15 com ácaro; predições
de `rocole_avaliar.py` sem deslocamento; IC 95% por bootstrap de plantas):

| Estratégia | Ferrugem: sensibilidade | Ferrugem: especificidade | Ácaro: sensibilidade (15 folhas) | Ácaro: especificidade | Folha certa |
|---|---|---|---|---|---|
| Só a face | 88,1% (37/42) | 91,2% | 13,3% (2/15) | 95,2% | 80,8% |
| Só o verso | 66,7% (28/42) | 94,7% | 60,0% (9/15) | 94,0% | 78,8% |
| **As duas fotos** | **92,9%** (39/42; 83,0% a 100%) | 89,5% (81,4% a 96,0%) | **60,0%** (9/15; 31,3% a 85,7%) | 90,5% (80,3% a 96,3%) | 80,8% |

- **As duas fotos acham mais doença do que só a face, como na validação:** ferrugem em 39 das 42 folhas (37 só com a
  face) e ácaro em 9 das 15 (2 só com a face). O custo são alarmes a mais: 6 folhas sem ferrugem chamadas de
  ferrugem (5 só com a face) e 8 sem ácaro chamadas de ácaro (4 só com a face). A fração de folhas certas nas duas
  doenças não muda (80,8%).
- **O ácaro fica abaixo da validação** (60% contra 80%), como já acontecia por foto (F1 do ácaro 0,465 no teste
  contra 0,727 na validação). No teste, todo o ácaro achado vem do verso: as duas fotos acham as mesmas 9 folhas que
  só o verso. O intervalo (31% a 86%) mostra o quanto 15 folhas pesam pouco.
- **Para o aplicativo:** pedir as duas fotos. Se for uma só, a escolha depende da doença: a face acha mais ferrugem
  (88% contra 67%), e o verso acha mais ácaro (60% contra 13%).

Arquivos: `outputs/resnet50/validacao_ajustes.json`, `folhas_val.csv`, `folhas_teste.csv`, `folhas_teste_ic.csv` e
os logs.

## Detector de ferrugem por regras (réplica de Rodriguez-Gallo et al., 2023)

Réplica do método de *Robust Coffee Rust Detection Using UAV-Based Aerial RGB Imagery* (Rodriguez-Gallo,
Escobar-Benitez e Rodriguez-Lainez, AgriEngineering 5, 2023), que relata 97% no RoCoLe sem rede neural:
variância local por canal (janela circular 5×5, como o filtro Variance do ImageJ) → máscara de vermelho/laranja →
contornos preenchidos → abertura e fechamento (elipse 5×5) → transformada de Hough de círculos → "ferrugem" se houver
pelo menos *k* círculos. Roda nas fotos originais 1920×1080, só com CPU (`src/rocole_regras.py`, OpenCV).

O artigo não publica os limiares de cor, os parâmetros do Hough nem a regra de decisão. Eles foram escolhidos no
treino, numa grade de 3 faixas de cor × 3 sensibilidades do Hough × 6 valores de *k*, pela acurácia balanceada
(escolhido: faixa estreita, `param2` 18, *k* = 1; treino 0,609, validação 0,585). A pergunta é binária: há
ferrugem na foto? (positivo = ferrugem; negativo = saudável e ácaro). O teste foi pontuado uma vez, nas mesmas 208
fotos da ResNet-50, cujas predições de 3 classes foram reduzidas a ferrugem/não ferrugem.

| Teste (208 fotos, 81 com ferrugem) | Regras | IC 95% | ResNet-50 | IC 95% |
|---|---|---|---|---|
| Sensibilidade (ferrugem encontrada) | 0,432 | 0,261 a 0,535 | 0,815 | 0,660 a 0,904 |
| Especificidade | 0,756 | 0,684 a 0,827 | 0,913 | 0,843 a 0,960 |
| **Acurácia balanceada** | **0,594** | 0,507 a 0,649 | **0,864** | 0,767 a 0,921 |
| F1 (ferrugem) | 0,476 | 0,322 a 0,559 | 0,835 | 0,700 a 0,906 |
| Acurácia | 63,0% | 55,3% a 68,4% | 87,5% | 78,9% a 92,5% |

Fração das fotos chamadas de "ferrugem": saudáveis 24,8% (ResNet-50: 1,9%), ácaro 22,7% (40,9%), ferrugem 43,2%
(81,5%).

- **Bem abaixo dos 97% do artigo, e da ResNet-50.** Diferenças de protocolo explicam parte: no artigo, especialistas
  revisaram só as fotos em que o método achou ferrugem e reclassificaram 96 "saudáveis" e 47 "ácaro" como
  ferrugem; os parâmetros foram ajustados e avaliados nas mesmas 1.560 fotos.
- **Alarmes fora da folha-alvo:** solo, folhas secas e folhas vizinhas geram anéis vermelhos/laranja e círculos
  (`outputs/regras/exemplos_detector.png`). O método pergunta "há ferrugem em algum lugar da foto", e os
  rótulos do RoCoLe falam da folha-alvo.
- **Perde lesões no verso da folha:** no teste, acha ferrugem em 57,8% das fotos da face (`H`) e em 25,0% das do
  verso (`E`). Manchas claras sobre o verso, que é mais pálido, não viram anéis vermelhos/laranja no mapa de
  variância. Os exemplos do artigo são da face.
- **Não separa ácaro de ferrugem o bastante** para ajudar a ResNet-50 nesse ponto: na face da folha, alarma em
  37,5% das fotos de ácaro e 57,8% das de ferrugem.
- O que vale do artigo: o método é simples, explicável e barato (CPU), e a observação de que fotos "saudáveis" e
  "ácaro" do RoCoLe às vezes mostram ferrugem em outra folha confirma o ruído de rótulo que vimos.

Arquivos em `outputs/regras/`: `parametros.json`, `ajuste_grade_treino.csv`, `contagens_ajuste.csv`,
`predicoes_teste.csv`, `resultados_teste.json`, `teste_intervalos_confianca.csv`, `etapas_detector.png`,
`exemplos_detector.png` e os logs.

## Recortar a folha-alvo (SAM)

A folha-alvo fica no centro da foto, com outras folhas e o solo em volta. Para testar se a CNN se sai melhor olhando
só para ela, a folha foi segmentada com o SAM (`facebook/sam-vit-base`, via `transformers`), usando um ponto no
centro da foto como pista (`src/rocole_folha.py`). Fica a parte da máscara ligada ao centro. Ela é aceita se cobrir
de 4% a 60% da foto, com confiança ≥ 0,75; senão, usa-se a foto inteira.

- **1.353 de 1.479 máscaras aceitas (91,5%).** As outras 126 usam a foto inteira (114 por baixa confiança, 12 sem
  máscara válida no centro). A aceitação é menor no ácaro (81% a 86% por split) que nas saudáveis (94% a 97%). A
  folha ocupa, em mediana, 24% da foto. Amostra em `outputs/segmentacao_amostra.jpg`.
- Duas entradas novas, também em 512×288: a folha recortada (caixa da máscara + 10% de margem) e a mesma com o fundo
  apagado.

**Comparação por validação cruzada** (`src/rocole_cv.py`): 3 dobras por planta no treino + validação (1.271 fotos; o
teste não entra), com a mesma receita de treino e a mesma seed por dobra para as três entradas. A regra foi fixada
antes de rodar: uma entrada recortada só substitui a foto inteira se o F1 macro médio for maior **e** se ela vencer
em pelo menos 2 das 3 dobras.

| Fora da dobra | Foto inteira | Folha recortada | Recorte com fundo apagado |
|---|---|---|---|
| F1 macro, dobra 1 | **0,818** | 0,790 | 0,787 |
| F1 macro, dobra 2 | **0,794** | 0,774 | 0,783 |
| F1 macro, dobra 3 | 0,738 | **0,829** | 0,787 |
| **F1 macro, média** | 0,783 | **0,798** | 0,786 |
| Recall do ácaro, média | 0,647 | 0,699 | 0,572 |
| Ácaro pela folha (duas fotos), sensibilidade média | 0,765 | 0,797 | 0,705 |

- **Fica a foto inteira.** O recorte tem a maior média (+0,015), mas a vantagem vem toda da dobra 3; vence em só 1
  das 3 dobras. O recorte com fundo apagado também vence só na dobra 3. Nenhuma passa a regra. A própria foto
  inteira varia de 0,738 a 0,818 entre dobras: uma diferença de 0,015 na média está dentro desse ruído.
- O recorte achou mais ácaro em duas dobras (recall 0,733 e 0,773, contra 0,644 e 0,659), mas não na primeira
  (0,591 contra 0,636): é um sinal, não uma prova. Apagar o fundo piora o ácaro (0,572). Uma causa possível: a
  máscara às vezes tem buracos sobre as lesões, e parte da lesão é apagada.
- Como a foto inteira ficou, o modelo final continua sendo `models/rocole_resnet50.pt`, e os números do teste acima
  não mudam. Os 9 modelos das dobras ficam em `models/cv/` (fora do git). Os resultados ficam em
  `outputs/cv_entradas/` (`resultados.csv`, `decisao.json`, `predicoes_fora_da_dobra.csv`, históricos e log).

**Detector por regras só dentro da folha** (treino e validação; 1.163 das 1.271 fotos com máscara, as outras com a
foto inteira). Com os parâmetros escolhidos de novo no treino (faixa larga, `param2` 18, *k* = 1), a acurácia
balanceada fica em 0,661 no treino e 0,680 na validação, contra 0,585 com a foto inteira: tirar o fundo corta boa
parte dos alarmes falsos. Mas o detector separa ainda pior o ácaro da ferrugem: na validação, chama de "ferrugem"
68,2% das fotos de ácaro e 59,3% das de ferrugem (saudáveis: 14,0%). Não ajuda onde a ResNet-50 erra, e por isso
não foi pontuado no teste. Arquivos em `outputs/regras_folha/`.

## Detector de ferrugem com a transformada de Hough

Pergunta: **há ferrugem na foto?** (positivo = ferrugem; negativo = saudável e ácaro), a mesma do detector por
regras. Lá o Hough é o *classificador* ("ferrugem se houver pelo menos *k* círculos") e o F1 fica em 0,476. Aqui o
Hough faz o que sabe fazer, achar **onde** há algo redondo, e um classificador treinado diz **o que** cada círculo é.

### A transformada de Hough, em poucas linhas

- É um esquema de **votação**. Cada pixel de borda vota em todas as formas que poderiam passar por ele; formas
  apoiadas por muitos pixels viram picos num acumulador. Para um círculo, os parâmetros são o centro (a, b) e o raio
  r: (x − a)² + (y − b)² = r².
- Por contar votos, tolera falhas, ruído e círculos incompletos, o que combina com lesões de contorno irregular.
- O `HOUGH_GRADIENT` do OpenCV evita o acumulador de 3 dimensões em duas etapas: as bordas do Canny votam em
  **centros** ao longo da direção do gradiente; depois cada centro recebe o raio mais apoiado pelas bordas em volta.
  `param1` é o limiar alto do Canny; `param2`, o mínimo de votos de um centro (menor = mais círculos, e mais fracos).
- Por que a regra "pelo menos *k* círculos" falha: solo, folhas secas, furos e dano de ácaro também são redondos. A
  contagem de círculos separa ferrugem do resto com AUC de no máximo 0,63 na foto inteira e 0,69 dentro da folha
  (melhor combinação da grade, em treino + validação; `contagens_ajuste.csv` de `outputs/regras/` e
  `outputs/regras_folha/`).

### Da foto aos atributos (`src/rocole_hough.py`)

1. **Contorno da folha.** A máscara do SAM (ponto no centro). Diferente da CNN, aqui a máscara de baixa confiança
   também é usada quando a área é válida (4% a 60% da foto): para o Hough, um contorno imperfeito é melhor que a foto
   inteira, com solo e folhas secas. A confiança do SAM vira um atributo. Sem máscara de área válida (12 das 1.479
   fotos), o contorno sai só da cor.
2. **Escala.** A folha é recortada e redimensionada para que a raiz da sua área seja 700 px. Os raios passam a valer
   em "unidades de folha", e o mesmo código serve a fotos de qualquer tamanho.
3. **Mapa de lesão.** Em L\*a\*b\*, o desvio do ângulo de matiz de cada pixel em relação ao verde da própria folha
   (o quarto mais verde dela), rumo a amarelo, laranja ou marrom, pesado pelo croma. Reflexos e sombras, quase sem
   cor, ficam perto de zero.
4. **Hough de círculos** sobre o mapa (`param1` 60, `param2` 12, raio de 4 a 48 px), permissivo de propósito: ficam
   até os 60 círculos de mais votos, e o classificador descarta os ruins. No treino, a mediana é de 5 a 12 círculos
   nas folhas saudáveis e de 56 a 60 nas doentes; 99% das fotos de ferrugem têm pelo menos um círculo
   (`outputs/hough/propostas_treino.jpg`).
5. **Extração de características:** 40 atributos por círculo e 27 pela folha inteira.

| Atributos | O que medem |
|---|---|
| Geometria e apoio do Hough: raio, distância à borda da folha, fração da circunferência com borda, ordem de votos | quão redondo e quão grande |
| Cor no disco: L\*a\*b\*, croma, desvios, valor do mapa de lesão | a cor da lesão: halo amarelo, pústula laranja, centro marrom |
| Contraste entre o disco e o anel em volta (ΔL, Δa\*, Δb\*) | quanto o círculo difere da folha ao redor |
| Perfil radial (miolo menos coroa) | centro escuro com halo, típico da ferrugem |
| Textura: variância local (o filtro do artigo), LBP uniforme (10 faixas), GLCM (contraste, homogeneidade, energia, correlação) | pústula pulverulenta contra folha lisa |
| Forma da mancha sob o círculo: área, circularidade, alongamento, solidez | mancha redonda contra risco ao longo das nervuras (ácaro) |
| Folha inteira: cor média e desvio, fração do mapa acima de 4 limiares ("% da folha infectada"), LBP, variância, confiança do SAM, área | clorose e severidade gerais |

LBP e GLCM foram escritos com NumPy e OpenCV; o scikit-image não entra nas dependências.

### Modelo de dois níveis (`src/rocole_hough_treinar.py`)

- **Nível 1, por círculo:** gradient boosting com rótulo fraco (cada círculo herda a classe da sua foto). Diz quanto
  o círculo parece ser de foto saudável, de ácaro ou de ferrugem.
- **Nível 2, por foto:** as pontuações dos círculos são resumidas (máximo, média dos 3 e dos 5 maiores, soma,
  contagens, área ponderada, e como são os 5 círculos mais parecidos com ferrugem) e juntadas aos atributos da folha.
  Uma floresta de árvores extremamente aleatórias dá as três probabilidades; a foto é "ferrugem" quando P(ferrugem)
  passa de um limiar.
- O nível 2 só vê pontuações de círculos feitas **fora da dobra**: nenhuma planta treina o nível 1 que a pontua.

### Validação cruzada

Treino + validação (1.271 fotos), nas mesmas 3 dobras por planta de `rocole_cv.py`; o teste não entra. Métrica: F1 de
ferrugem foto a foto, fora da dobra, com o limiar de cada dobra escolhido nas outras duas. Regra fixada antes de
rodar (a mesma dos recortes do SAM): uma variante só substitui a base se a média for maior **e** se vencer em pelo
menos 2 das 3 dobras.

| Fora da dobra, F1 de ferrugem | Dobra 1 | Dobra 2 | Dobra 3 | Média |
|---|---|---|---|---|
| Base: círculos + folha, gradient boosting | 0,831 | 0,813 | 0,805 | 0,816 |
| Só a folha (**sem o Hough**) | 0,762 | 0,747 | 0,748 | 0,752 |
| Só os círculos | 0,775 | 0,810 | 0,785 | 0,790 |
| Base com regressão logística no nível 2 | 0,823 | 0,834 | 0,794 | 0,817 |
| Base com a média de boosting e logística | 0,831 | 0,823 | 0,804 | 0,819 |
| Base com floresta | 0,822 | 0,822 | 0,813 | 0,819 |
| Atributos ricos, boosting | 0,815 | 0,811 | 0,817 | 0,814 |
| Atributos ricos, média | 0,809 | 0,808 | 0,793 | 0,803 |
| **Atributos ricos, floresta (escolhida)** | 0,832 | 0,831 | 0,808 | **0,824** |
| Base com Hough mais rígido (`param1` 100, `param2` 20) | 0,749 | 0,784 | 0,721 | 0,751 |
| ResNet-50, foto inteira (referência, de `rocole_cv.py`) | 0,867 | 0,847 | 0,761 | 0,825 |

- **O Hough não é enfeite:** sem os círculos, o mesmo classificador cai de 0,816 para 0,752.
- **Todas as variantes clássicas ficam entre 0,80 e 0,82**, no nível da ResNet-50 (0,825) e mais estáveis entre
  dobras (0,808 a 0,832 na escolhida, contra 0,761 a 0,867). A escolhida passou a regra por margem pequena; com 10
  variantes comparadas, o 0,824 é levemente otimista.
- Menos círculos (Hough mais rígido) piora: 0,751.

**Validação cruzada repetida (auditoria).** A tabela acima usa uma única divisão em 3 dobras, e as variantes diferem
menos do que a oscilação entre dobras. Para ver se a ordem se mantém, a validação cruzada foi repetida com outras 4
divisões por planta: 5 repetições × 3 dobras (`rocole_hough_treinar.py cv-repetida`). É uma auditoria: o modelo
escolhido e os números do teste não mudam.

| F1 de ferrugem, 5 repetições | Média | Menor e maior repetição | Diferença para a base | Repetições em que vence a base |
|---|---|---|---|---|
| **Atributos ricos, floresta (escolhida)** | **0,807** | 0,788 a 0,824 | +0,013 | 5 de 5 |
| Atributos ricos, boosting | 0,807 | 0,804 a 0,814 | +0,013 | 4 de 5 |
| Atributos ricos, média | 0,806 | 0,797 a 0,823 | +0,011 | 4 de 5 |
| Base com a média de boosting e logística | 0,800 | 0,787 a 0,819 | +0,006 | 5 de 5 |
| Base com floresta | 0,799 | 0,788 a 0,819 | +0,004 | 5 de 5 |
| Base: círculos + folha, gradient boosting | 0,795 | 0,785 a 0,816 | | |
| Base com regressão logística | 0,789 | 0,771 a 0,817 | −0,006 | 2 de 5 |
| Só os círculos | 0,777 | 0,770 a 0,790 | −0,018 | 0 de 5 |
| Só a folha (**sem o Hough**) | 0,751 | 0,727 a 0,760 | −0,044 | 0 de 5 |

- **A escolha se mantém:** a variante escolhida fica no topo e vence a base nas 5 repetições. As três de atributos
  ricos empatam entre si (0,806 a 0,807).
- **A divisão original foi a mais favorável:** nela a escolhida deu 0,824; na média das 5, **0,807**. Esse é o número
  mais prudente para o modelo de Hough sozinho.
- **Sem os círculos do Hough, −0,044 em toda repetição.**
- A fusão não entra na auditoria: a ResNet-50 só tem modelos treinados para as 3 dobras originais (repetir exigiria
  treinar outras 12 redes).

### Fusão tardia com a ResNet-50

Como o modelo clássico ficou abaixo da meta, a resposta final passou a ser a **média ponderada das probabilidades**
do modelo de Hough e da ResNet-50: P = w · P(Hough) + (1 − w) · P(ResNet-50). Nenhum modelo é ajustado sobre as
saídas da ResNet-50. Fora da dobra, cada foto é prevista pela ResNet-50 da sua dobra (`models/cv/`), que não viu a
planta dela (as classes batem 100% com as gravadas por `rocole_cv.py`).

| Fora da dobra, F1 de ferrugem | Dobra 1 | Dobra 2 | Dobra 3 | Média |
|---|---|---|---|---|
| Modelo de Hough (w = 1) | 0,832 | 0,831 | 0,808 | 0,824 |
| Fusão, w = 0,75 | 0,856 | 0,839 | 0,822 | 0,839 |
| **Fusão, w = 0,50 (adotada)** | 0,867 | 0,855 | 0,820 | **0,847** |
| Fusão, w = 0,25 | 0,868 | 0,855 | 0,799 | 0,840 |
| ResNet-50 com limiar (w = 0) | 0,850 | 0,854 | 0,798 | 0,834 |

- **Adotada com w = 0,50 e limiar 0,28:** vence o modelo de Hough nas 3 dobras. Com esse limiar único, o F1 fora da
  dobra é 0,859 (sensibilidade 0,918, especificidade 0,864).
- A fusão também fica acima da ResNet-50 sozinha (0,834, já com limiar ajustado): os dois modelos erram em fotos
  diferentes.

### Teste

Uma vez por modelo (`src/rocole_hough_avaliar.py` para se o resultado já existe): 208 fotos, 81 com ferrugem, nas
mesmas fotos do detector por regras e da ResNet-50. IC 95% por bootstrap de plantas. Configuração, peso da fusão e
limiares vieram só de treino + validação.

| Ferrugem, foto a foto | Fusão | Modelo de Hough | ResNet-50 | Regras |
|---|---|---|---|---|
| Sensibilidade | 0,938 (0,855 a 0,987) | 0,963 (0,902 a 1,000) | 0,815 (0,660 a 0,904) | 0,432 (0,261 a 0,535) |
| Especificidade | 0,874 (0,784 a 0,935) | 0,850 (0,734 a 0,927) | 0,913 (0,843 a 0,960) | 0,756 (0,684 a 0,827) |
| Acurácia balanceada | 0,906 (0,837 a 0,949) | 0,907 (0,828 a 0,954) | 0,864 (0,767 a 0,921) | 0,594 (0,507 a 0,649) |
| **F1** | **0,879** (0,774 a 0,934) | 0,876 (0,761 a 0,941) | 0,835 (0,700 a 0,906) | 0,476 (0,322 a 0,559) |
| Acurácia | 89,9% (82,2% a 94,4%) | 89,4% (80,3% a 95,0%) | 87,5% (78,9% a 92,5%) | 63,0% (55,3% a 68,4%) |

Fração das fotos chamadas de "ferrugem", por classe: saudáveis 1,0% (fusão), 3,8% (Hough), 1,9% (ResNet-50); ácaro
68,2%, 68,2% e 40,9%; ferrugem 93,8%, 96,3% e 81,5%.

- **Fusão: F1 0,879.** Acha 76 das 81 fotos de ferrugem e dá 16 alarmes em 127 fotos sem ferrugem. O modelo de Hough
  sozinho fica em 0,876 (78 de 81; 19 alarmes). Os dois ficam acima da ResNet-50 (0,835), mas os intervalos se
  sobrepõem: são 208 fotos.
- **Os alarmes vêm do ácaro:** 15 das 22 fotos de ácaro são chamadas de ferrugem, contra 1 das 105 saudáveis. O
  modelo separa bem folha doente de folha sadia e mal ácaro de ferrugem. Nas 3 classes, o F1 macro da fusão é 0,752
  (IC 0,648 a 0,827; ácaro 0,400), o mesmo nível da ResNet-50 (0,753).
- **O verso deixou de ser o ponto cego:** a fusão acha ferrugem em 91,7% das fotos do verso e 95,6% das da face
  (regras: 25,0% e 57,8%; ResNet-50: 75,0% e 86,7%).
- **O teste ficou acima da validação cruzada** (0,879 contra 0,847), como já tinha acontecido ao contrário com a
  ResNet-50: uma única divisão oscila. O número mais prudente é o de fora da dobra.

**Pela folha inteira** (99 folhas com as duas fotos, 42 com ferrugem; a ferrugem existe se aparece em algum lado).
F1 de ferrugem, com acertos e alarmes entre parênteses (`outputs/hough/folhas_teste_f1.csv`, calculado das
predições já gravadas):

| F1 de ferrugem por folha | Só a face | Só o verso | As duas fotos |
|---|---|---|---|
| Fusão | 0,909 (40 de 42; 6 alarmes) | 0,843 (35; 6) | **0,901** (41 de 42; 8 alarmes) |
| Modelo de Hough | 0,911 (41; 7) | 0,837 (36; 8) | 0,872 (41; 11) |
| ResNet-50 | 0,881 (37; 5) | 0,767 (28; 3) | 0,897 (39; 6) |

Com as duas fotos, a fusão tem sensibilidade de 97,6% (IC 91,4% a 100%) e especificidade de 86,0% (76,2% a 92,7%).

### Opção de recusa ("mostre a um agrônomo")

Como o modelo de 4 doenças do BRACOL, o detector pode **não responder** quando está em dúvida: a foto é
recusada se P(ferrugem) cai numa faixa em volta do limiar (`src/rocole_hough_recusa.py`). Regra fixada antes de
rodar: fica a **menor** taxa de recusa cujo F1 de ferrugem nas fotos aceitas chega a 0,90, fora da dobra, com o
limiar e a faixa de cada dobra escolhidos nas outras duas.

| Recusa, fora da dobra | Fotos aceitas | F1 nas aceitas | Menor e maior dobra |
|---|---|---|---|
| 0% | 100% | 0,847 | 0,820 a 0,867 |
| 5% | 93,6% | 0,875 | 0,860 a 0,890 |
| 10% | 88,3% | 0,887 | 0,878 a 0,905 |
| 15% | 84,2% | 0,899 | 0,890 a 0,917 |
| **20% (escolhida)** | 79,3% | **0,911** | 0,900 a 0,923 |
| 25% | 75,1% | 0,921 | 0,919 a 0,922 |
| 30% | 70,1% | 0,934 | 0,929 a 0,944 |

- **Escolhida: recusar 20%**, o que dá a faixa de 0,131 a 0,429 em volta do limiar 0,28. As três dobras ficam em 0,90
  ou mais.
- **No teste, uma vez** (das probabilidades já gravadas, sem rodar modelo): o detector aceita 179 das 208 fotos
  (86,1%; IC 95% 75,6% a 92,6%) e, nelas, o F1 de ferrugem é **0,935** (IC 0,845 a 0,980): acha 72 das 73 fotos de
  ferrugem e dá 9 alarmes em 106 sem ferrugem. Sem recusa, 0,879.
- **As 29 recusadas** são 11 das 22 fotos de ácaro, 8 das 81 de ferrugem e 10 das 105 saudáveis: a recusa tira
  metade do ácaro, que é onde o modelo erra.
- Antes de fixar a regra eu já tinha visto a curva de recusa fora da dobra (sem aninhar); o teste não tinha sido
  olhado.

### A meta de 0,90

- **Foto a foto, respondendo a todas, não foi atingida:** 0,879 no teste (IC 0,774 a 0,934) e 0,847 fora da dobra.
- **Com a opção de recusa, é atingida nas fotos que o detector aceita responder:** 0,911 fora da dobra (79% das
  fotos) e 0,935 no teste (86%). O preço é mandar ao agrônomo de uma em cada sete a uma em cada cinco fotos.
- **Pela folha inteira, com as duas fotos, o teste dá 0,901.** É outra unidade (99 folhas), não era a métrica fixada
  antes de rodar, e o intervalo é largo: vale como sinal, não como meta cumprida.
- O que segura o número: ácaro contra ferrugem (155 fotos de ácaro, com ruído de rótulo conhecido entre as duas
  classes) e a oscilação de uma divisão pequena.

### Em que folhas o detector foi medido

O detector das seções acima (a fusão) foi **treinado só no RoCoLe** e medido em dois conjuntos, de duas espécies,
dois países e dois tipos de foto. O modelo treinado no BRACOL está na
[seção seguinte](#treino-e-teste-por-conjunto---dataset).

| | RoCoLe | BRACOL |
|---|---|---|
| Espécie e origem | robusta; uma lavoura (autores do Equador) | arábica; Espírito Santo, Brasil |
| Tipo de foto | folha na planta, fundo de campo | folha destacada, fundo claro |
| Papel | treino, validação cruzada e **teste** (208 fotos de plantas nunca vistas) | **conferência externa** (598 folhas); nada treinado nem ajustado |
| Ferrugem achada | 76 de 81 fotos (93,8%) | 304 de 304 folhas (100%) |
| Saudáveis chamadas de ferrugem | 1 de 105 (1,0%) | 20 de 144 (13,9%) |
| Outras lesões chamadas de ferrugem | ácaro: 15 de 22 (68,2%) | bicho-mineiro, phoma ou cercosporiose: 132 de 150 (88,0%) |

Números da fusão, respondendo a todas as fotos (sem a recusa).

- **Para achar ferrugem, o detector funcionou nos dois conjuntos:** não dependeu da espécie, da região nem do tipo de
  foto.
- **O que muda fora do RoCoLe são os alarmes:** mais folhas saudáveis de arábica saem como ferrugem, e lesões de outras
  doenças quase sempre saem como ferrugem. O detector só conhece saudável, ácaro e ferrugem.
- **O que não foi medido:** arábica fotografada na planta (os dois conjuntos não têm essa combinação) e folhas de
  outras regiões além dessas duas.

### Treino e teste por conjunto (`--dataset`)

Os comandos de treino e de teste aceitam `--dataset rocole|bracol` (ou os atalhos `--rocole` e `--bracol`). Sem a
opção, é o RoCoLe, e nada do que está acima muda. O código é o mesmo: só mudam as fotos, as classes e a divisão
(`src/rocole_hough_dados.py`). **Cada conjunto tem o seu modelo, treinado e testado nele mesmo; os dois não são
juntados.**

| Treino | Teste | Modelo gravado | Resultados |
|---|---|---|---|
| RoCoLe (treino + validação, 1.271 fotos) | RoCoLe (208 fotos) | `models/rocole_hough.joblib` | `outputs/hough/` |
| BRACOL (treino + validação, 1.200 folhas) | BRACOL (201 folhas) | `models/bracol_hough.joblib` | `outputs/hough_bracol/` |

```bash
python -P src/rocole_hough_treinar.py mascaras --bracol   # SAM acha a folha em cada foto (GPU)
python -P src/rocole_hough_treinar.py extrair --bracol    # circulos do Hough e atributos
python -P src/rocole_hough_treinar.py cv --bracol         # validacao cruzada; resultados em outputs/hough_bracol/
python -P src/rocole_hough_treinar.py final --bracol      # grava models/bracol_hough.joblib
python -P src/rocole_hough_avaliar.py --bracol            # teste do BRACOL com o modelo do BRACOL, uma vez
```

- **BRACOL:** 1.401 folhas de arábica, com a divisão do modelo de 4 doenças (1.000 / 200 / 201, agrupada por pHash).
  "Ferrugem" é toda folha com `rust = 1`, sozinha ou com outra doença; as classes são saudável (144 folhas), outra
  doença sem ferrugem (652) e ferrugem (605). O SAM achou a folha nas 1.401 fotos.
- No BRACOL é o **modelo de Hough sozinho**. A fusão com a ResNet-50 e a recusa existem só no RoCoLe (não há rede
  treinada por dobra para as classes do BRACOL).

Validação cruzada (F1 de ferrugem fora da dobra; mesmas variantes e mesma regra de escolha nos dois conjuntos):

| Fora da dobra | RoCoLe | BRACOL |
|---|---|---|
| Base: círculos + folha | 0,816 | 0,919 |
| Só a folha (**sem o Hough**) | 0,752 | 0,870 |
| Variante escolhida | 0,824 (atributos ricos, floresta) | 0,939 (atributos ricos, média) |

Teste, uma vez por modelo (F1 de ferrugem por foto; IC 95% por bootstrap de grupos):

| | Treino RoCoLe → teste RoCoLe | Treino BRACOL → teste BRACOL |
|---|---|---|
| Fotos de teste (com ferrugem) | 208 (81) | 201 (86) |
| **Modelo de Hough: F1** | **0,876** (0,761 a 0,941) | **0,949** (0,912 a 0,979) |
| Sensibilidade | 0,963 (78 de 81) | 0,965 (83 de 86) |
| Especificidade | 0,850 (19 alarmes em 127) | 0,948 (6 alarmes em 115) |
| Saudáveis chamadas de ferrugem | 4 de 105 (3,8%) | 0 de 21 |
| Outra lesão chamada de ferrugem | ácaro: 15 de 22 (68,2%) | outras doenças: 6 de 94 (6,4%) |
| Fusão com a ResNet-50: F1 | 0,879 (0,774 a 0,934) | não há |
| Fusão com recusa: F1 nas fotos aceitas | 0,935 (0,845 a 0,980), 86% das fotos | não há |

- **No BRACOL o modelo passa de 0,90 sem fusão e sem recusa:** 0,949 no teste e 0,939 fora da dobra. Treinado com as
  outras doenças, ele as separa da ferrugem: só 6 das 94 folhas com outra doença saem como ferrugem, e nenhuma
  saudável.
- **No RoCoLe fica abaixo de 0,90 respondendo a todas as fotos** (0,876 sozinho, 0,879 na fusão). O erro é o ácaro:
  15 das 22 fotos de ácaro saem como ferrugem. Com a recusa, chega a 0,935 nas fotos aceitas.
- **O Hough não é enfeite em nenhum dos dois:** sem os círculos, a base perde 0,064 no RoCoLe e 0,049 no BRACOL.
- **Os dois números não se comparam diretamente.** São conjuntos diferentes: no BRACOL a folha está destacada, sobre
  fundo claro, e a "outra lesão" são doenças com muitas fotos; no RoCoLe a folha está na planta e a "outra lesão" é o
  ácaro, com 155 fotos e ruído de rótulo.

Arquivos do BRACOL em `outputs/hough_bracol/`: `cv_resumo.csv`, `cv_oof.csv`, `decisao.json`, `resultados_teste.json`,
`predicoes_teste.csv`, `teste_intervalos_confianca.csv`, `folhas_sam.csv`, `propostas_treino.jpg`, `app_exemplos.jpg`
e os logs.

### Conferência externa: folhas do BRACOL

Para saber o que esperar do modelo do RoCoLe fora do RoCoLe, o modelo pronto (fusão, com o SAM achando a folha na hora)
rodou em folhas do BRACOL: café **arábica**, folha destacada sobre fundo claro
(`rocole_hough_avaliar.py --conferencia-externa`). Foi feita antes de existir um modelo treinado no BRACOL.
**Nada foi ajustado nessas fotos.**

| Grupo do BRACOL | Folhas | Chamadas de ferrugem | IC 95% |
|---|---|---|---|
| Só ferrugem | 304 | **304 (100%)** | 98,8% a 100% |
| Saudáveis | 144 | 20 (13,9%) | 9,2% a 20,5% |
| Outra doença, sem ferrugem (amostra de 150) | 150 | 132 (88,0%) | 81,8% a 92,3% |

- **Acha a ferrugem em outra espécie e em outro tipo de foto**, em todas as severidades (202 das 304 folhas são de
  severidade 1, a mais baixa). O SAM achou a folha em 100% das fotos.
- **Mas chama de ferrugem quase toda folha com lesão:** 88% das folhas com bicho-mineiro, phoma ou cercosporiose. O
  modelo só conhece saudável, ácaro e ferrugem; fora disso, "tem lesão" vira "ferrugem". E alarma em 13,9% das
  saudáveis de arábica (1% no teste do RoCoLe).
- Leitura: fora do RoCoLe, o modelo responde "a folha tem lesão?", não "a lesão é ferrugem?".
- Com a faixa de recusa aplicada às mesmas probabilidades (`bracol_externo.csv`): nenhuma das 304 folhas com
  ferrugem fica incerta; das 144 saudáveis, 40 ficam incertas e 14 continuam "ferrugem"; das 150 com outra doença, 53
  ficam incertas e 96 continuam "ferrugem". A recusa reduz os alarmes, mas não resolve as outras doenças.

Arquivos em `outputs/hough/`: `cv_resumo.csv`, `cv_oof.csv`, `decisao.json`, `fusao_resumo.csv`, `fusao_oof.csv`,
`resnet50_oof.csv`, `resultados_teste.json` e `resultados_teste_fusao.json` (mais predições, IC, folhas e exemplos de
cada um), `folhas_teste_f1.csv`, `bracol_resumo.csv`, `bracol_externo.csv`, `bracol_exemplos.jpg`,
`propostas_treino.jpg`, `cv_repetida.csv` e `cv_repetida_resumo.csv`, `recusa_cv.csv`, `recusa_resumo.csv`,
`recusa_teste.json` (mais IC e predições) e os logs.

## Aplicativo

`app/app.py` é uma página local (Flask) que aplica o detector a fotos soltas, **sem máscara**: o contorno da folha é
achado na hora pelo SAM, com um ponto no centro da foto (`src/rocole_hough_inferir.py`).

O objetivo é detectar **ferrugem em qualquer folha de café**, independente da região, da espécie e do tipo de foto.
Por isso o aplicativo não pede máscara nem um tamanho de foto e mede as lesões em relação ao tamanho da folha. O que
já foi medido, e o que falta para chegar a "qualquer folha", está em [Limitações](#limitações).

```bash
python -P app/app.py                   # modelo do BRACOL (padrao); depois abra http://127.0.0.1:5000
python -P app/app.py --modelo rocole   # modelo do RoCoLe: fusao com a ResNet-50 e resposta "Incerto"
```

- **O modelo padrão é o treinado no BRACOL** (`models/bracol_hough.joblib`): o modelo de Hough sozinho. É o de F1
  mais alto no próprio teste (0,949) e o que separa a ferrugem das outras doenças. Mostra as classes Saudável, Outra
  lesão e Ferrugem, e não tem a resposta "Incerto".
- **`--modelo rocole`** usa o detector do RoCoLe (`models/rocole_hough.joblib`): a fusão com a ResNet-50 e a opção de
  recusa. Mostra Saudável, Ácaro-vermelho e Ferrugem. **"Incerto"** sai quando a probabilidade de ferrugem fica entre
  13% e 43%, e a página pede para mostrar a folha a um agrônomo.
- O rodapé da página diz em que fotos o modelo carregado foi treinado. `--porta` troca a porta.
- Envie uma ou mais fotos (JPG ou PNG, até 12 por vez), com a folha no centro. Fotos de qualquer tamanho e
  orientação; a rotação do celular (EXIF) é corrigida.
- Para cada foto: o veredito, as três probabilidades e a foto com o contorno da folha e os círculos do Hough, do
  azul (não parece ferrugem) ao vermelho (parece). Com `--modelo rocole`, também a parte de cada modelo na fusão. Um
  aviso aparece quando o SAM não acha a folha (o contorno sai da cor) ou quando nenhum círculo é encontrado.
- Os envios são analisados um por vez: dois cliques seguidos em "Analisar" esperam na fila (antes, chegavam juntos
  ao modelo e derrubavam a carga do SAM).
- Precisa do modelo escolhido em `models/` e do SAM (~375 MB, baixado na primeira vez). Com `--modelo rocole`,
  também dos pesos `models/rocole_resnet50.pt` (sem eles, responde só o modelo de Hough, com o limiar
  dele, e avisa). `--sem-sam` e `--sem-resnet` desligam essas partes.
- Tempos medidos com o modelo do RoCoLe numa RTX 4050: a primeira análise carrega os modelos (27 s para 4 fotos);
  depois, cerca de 1,5 s por foto. O modelo do BRACOL não carrega a ResNet-50. Sem GPU deve rodar, mais devagar (não
  foi testado). Só atende a própria máquina.
- **Conferido com o modelo padrão**, pela rota da página, em 12 fotos dos conjuntos de teste sorteadas com semente
  fixa (2 por classe de cada conjunto). As 6 do BRACOL saíram certas (2 com ferrugem, 2 com outra doença, 2
  saudáveis). Das 6 do RoCoLe, fotos de campo que esse modelo nunca viu: as 2 com ferrugem saíram "Ferrugem", as 2
  de ácaro saíram "Sem ferrugem" (como "Outra lesão") e **1 das 2 saudáveis saiu "Ferrugem"** (81%). Exemplo em
  `outputs/hough_bracol/app_exemplos.jpg`. São 12 fotos: servem para ver o aplicativo funcionando, não para medir.
- Conferido com `--modelo rocole`: duas folhas de ferrugem do BRACOL saíram "Ferrugem" (97% e 96%), uma saudável
  saiu "Sem ferrugem" (6%), uma imagem que não é folha saiu "Sem ferrugem" com aviso, e um arquivo de texto foi
  recusado com mensagem. Uma folha saudável do BRACOL com 24% de probabilidade saiu "Incerto", e dois envios ao
  mesmo tempo responderam os dois. Exemplos da página em `outputs/hough/app_exemplo.jpg` e `app_incerto.jpg`.

> O aplicativo é um **apoio à decisão**. Cada modelo foi medido só no tipo de foto em que foi treinado; em outro
> tipo de foto, o que se viu foram mais alarmes. Um "Ferrugem" pede conferência de um agrônomo.

## Limitações

**Dados e medição**

- **Dois conjuntos, cada um de uma origem só.** RoCoLe: café robusta de uma lavoura, fotografado na planta. BRACOL:
  café arábica do Espírito Santo, folha destacada sobre fundo claro. Arábica fotografada na planta, robusta destacada
  e folhas de outras regiões não foram medidas.
- **Cada modelo foi treinado e testado no próprio conjunto, e os dois não foram juntados.** Não há um número que
  valha para "qualquer folha de café". O resultado de um conjunto não se transfere para o outro: na conferência
  externa, o detector do RoCoLe achou a ferrugem de todas as 304 folhas do BRACOL, mas chamou de ferrugem 13,9% das
  saudáveis (1,0% no RoCoLe) e 88% das que têm outra doença. No sentido contrário (modelo do BRACOL em fotos de
  campo) só há a conferência do aplicativo com 6 fotos, em que 1 das 2 folhas saudáveis saiu como ferrugem.
- **Testes pequenos e uma divisão só.** 208 fotos no RoCoLe (81 com ferrugem, 22 de ácaro) e 201 folhas no BRACOL (86
  com ferrugem, 21 saudáveis). Os intervalos são largos, principalmente no RoCoLe (0,774 a 0,934 na fusão).
- **Rótulos.** O RoCoLe tem ruído conhecido (13 cópias com rótulos conflitantes) e talvez outros que a comparação de
  imagens não detecta. Em nenhum dos dois conjuntos há rótulo por lesão: cada círculo herda a classe da foto, então
  um círculo sobre uma parte sadia de uma folha com ferrugem conta como ferrugem no treino.
- **Sem severidade.** Os níveis de ferrugem não vêm na versão do RoCoLe do Kaggle, e no BRACOL este detector só
  responde se há ferrugem (a severidade fica com o modelo de 4 doenças).

**Modelo**

- **A meta de 0,90 no RoCoLe só é atingida com a opção de recusa**, isto é, sem responder a cerca de 20% das fotos.
  Respondendo a todas, fica em 0,879 no teste e 0,847 fora da dobra. As 10 variantes do modelo de Hough e os 3 pesos
  da fusão foram comparados nas mesmas 3 dobras, o que deixa os números de fora da dobra levemente otimistas: na
  média de 5 divisões, o modelo de Hough sozinho cai de 0,824 para 0,807.
- **Ácaro e ferrugem não são separados no RoCoLe:** 15 das 22 fotos de ácaro do teste saem como "ferrugem". São só
  155 fotos de ácaro, com ruído de rótulo entre as duas classes.
- **No BRACOL não há fusão nem recusa.** O modelo responde a todas as folhas, sem "Incerto". "Ferrugem" inclui as
  folhas com ferrugem e outra doença junto, e o modelo não diz qual é a outra doença.
- **O Hough procura círculos.** Lesões grandes que se juntaram, lesões alongadas (como a mina do bicho-mineiro) e
  lesões na borda da folha só podem ser aproximadas por vários círculos; quanto se perde com isso não foi medido em
  separado. São no máximo 60 círculos por folha, com raio de 4 a 48 px numa folha levada a 700 px de lado: pontos
  menores que isso não viram candidato.
- **Tudo depende do contorno da folha.** O SAM recebe um ponto no centro da foto. Se a folha não está no centro, ou
  se há várias folhas, o contorno sai errado e os atributos também. Quando o SAM não acha a folha, o contorno vem da
  cor, que é pior com fundo de campo.

**Aplicativo**

- **O modelo padrão foi treinado em folhas destacadas sobre fundo claro.** Em fotos tiradas na planta o esperado são
  mais alarmes (1 das 2 folhas saudáveis de campo da conferência saiu como ferrugem), e esse uso não foi medido além
  dessas 6 fotos. O objetivo de servir a qualquer folha ainda não foi demonstrado.
- Com `--modelo rocole`, a foto é deitada e cortada no centro em 16:9 para a ResNet-50, como no treino; fotos muito
  diferentes disso perdem parte da folha nessa metade da fusão.
- Roda só na própria máquina, uma foto por vez, e foi testado só com GPU (RTX 4050).

## Trabalhos futuros

- **Um modelo único para qualquer folha.** Juntar RoCoLe, BRACOL e outros conjuntos num só treino, e medir deixando
  um conjunto inteiro de fora a cada vez. Neste trabalho cada conjunto ficou com o seu modelo.
- **Fotos do uso real.** Coletar e rotular arábica fotografada na planta, de outras lavouras e regiões, para medir o
  aplicativo onde ele será usado e escolher o limiar para esse tipo de foto.
- **Fusão e recusa no BRACOL.** Treinar uma rede por dobra com as classes do BRACOL para ter, nesse conjunto, a
  mesma fusão e a resposta "Incerto" do RoCoLe.
- **Ácaro contra ferrugem.** Mais fotos de ácaro, revisão dos rótulos em conflito e atributos próprios do verso da
  folha, onde a ferrugem esporula.
- **Rótulo por lesão e severidade.** Anotar círculos de algumas centenas de folhas para treinar o primeiro nível com
  rótulo de verdade, e estimar a severidade pela fração da folha coberta por círculos de ferrugem.
- **Aplicativo.** Versão para celular e sem GPU (um segmentador mais leve que o SAM), várias folhas por foto e
  diagnóstico pela planta, juntando as fotos de várias folhas.

## Estrutura

Caminhos relativos à raiz do repositório. Os dados ficam em `data/`.

| Caminho | Conteúdo |
|---|---|
| `src/rocole_comum.py` | caminhos e constantes |
| `src/rocole_preparar.py` | remove cópias, cache 512×288, divisão agrupada por planta |
| `src/rocole_modelo.py`, `src/rocole_treinar.py` | dataset, ResNet-50 (ou ConvNeXt-Tiny), treino |
| `src/rocole_avaliar.py`, `src/rocole_gradcam.py` | métricas, IC por planta, matriz de confusão, Grad-CAM |
| `src/rocole_validacao.py` | deslocamentos por classe e diagnóstico pela folha inteira (escolhidos na validação; `--teste` pontua a folha inteira no teste) |
| `src/rocole_regras.py` | detector de ferrugem por regras (réplica de Rodriguez-Gallo et al., 2023) |
| `src/rocole_folha.py` | segmenta a folha-alvo com o SAM e cria os caches recortados |
| `src/rocole_cv.py` | validação cruzada por planta para comparar foto inteira e recortes |
| `src/rocole_hough.py` | da foto aos círculos do Hough e aos atributos (contorno, escala, mapa de lesão, LBP, GLCM) |
| `src/rocole_hough_treinar.py` | extração em lote, modelo de dois níveis, validação cruzada, fusão com a ResNet-50, modelo final |
| `src/rocole_hough_avaliar.py` | teste de cada conjunto com o modelo dele (uma vez por modelo), folha inteira e conferência externa no BRACOL |
| `src/rocole_hough_inferir.py` | aplica o modelo a uma foto solta, sem máscara (SAM na hora) |
| `src/rocole_hough_recusa.py` | opção de recusa: escolhe a taxa fora da dobra e pontua no teste, uma vez |
| `src/rocole_hough_dados.py` | os conjuntos que o detector sabe usar (`--dataset rocole` ou `bracol`): fotos, classes, divisão, pastas |
| `app/app.py`, `app/templates/` | aplicativo local (Flask) |
| `tests/` | testes (`unittest`) |
| `models/rocole_resnet50.pt` | pesos do modelo final |
| `models/rocole_resnet50.json` | SHA-256 do `metadados.csv` usado no treino |
| `models/cv/` | pesos dos 9 modelos da validação cruzada (fora do git) |
| `models/rocole_hough.joblib` | modelo de Hough (dois níveis), peso e limiar da fusão, faixa de recusa |
| `models/bracol_hough.joblib` | o mesmo modelo de Hough treinado no BRACOL (o padrão do aplicativo) |
| `outputs/hough/`, `outputs/hough_bracol/` | validação cruzada e teste do detector com o Hough: RoCoLe e BRACOL |
| `outputs/` | divisão, descartes, `comparacao_modelos.csv`, segmentação e uma pasta por etapa |
| `docs/specs/`, `docs/plans/` | desenhos e planos: os de 2026-11-11 (ResNet-50, escritos antes da mudança para esta pasta, com os caminhos antigos `src/` e `outputs/rocole/`) e os de 2026-11-13 (Hough e aplicativo) |
| `data/archive.zip` | zip do Kaggle (fora do git) |
| `data/cache_rocole_*`, `data/mascaras_rocole/` | caches 512×288 e máscaras do SAM (gerados, fora do git) |
| `data/cache_rocole_hough/` | atributos dos círculos e máscaras de baixa confiança (gerados, fora do git) |

## Como reproduzir

Requer Python 3.13 e, de preferência, GPU NVIDIA. As dependências estão em `requirements.txt`, que inclui
OpenCV, `transformers` e Flask.

Coloque o `archive.zip` do Kaggle (pastas `coffee___healthy`, `coffee___red_spider_mite`, `coffee___rust`) em
`data/archive.zip`, na raiz do repositório, ou passe `--zip CAMINHO` à preparação, às regras e ao SAM. Os comandos
rodam da raiz do repositório (no PowerShell: `$env:PYTHONPATH = "src"`):

```bash
export PYTHONPATH=src
python -P src/rocole_preparar.py         # remove copias, cria o cache 512x288 e divide por planta
python -P src/rocole_treinar.py          # ResNet-50; grava models/rocole_resnet50.pt (--modelo convnext_tiny)
python -P src/rocole_avaliar.py          # metricas, IC e Grad-CAM em outputs/resnet50/ (pontua o teste)
python -P src/rocole_validacao.py        # deslocamentos por classe e folha inteira (so validacao)
python -P src/rocole_validacao.py --teste  # folha inteira no teste, uma vez, com o modelo final
python -P src/rocole_regras.py ajustar   # detector por regras: escolhe parametros no treino (CPU, ~2 min)
python -P src/rocole_regras.py testar    # pontua o teste uma vez e compara com a ResNet-50
python -P src/rocole_folha.py            # SAM: mascaras da folha-alvo e caches recortados (~25 min; baixa ~375 MB)
python -P src/rocole_regras.py ajustar --mascara  # regras so dentro da folha (treino e validacao)
python -P src/rocole_cv.py               # 3 dobras x 3 entradas (~1h45 numa RTX 4050; retoma se parar)
python -P src/rocole_hough_treinar.py mascaras  # SAM nas fotos que rocole_folha.py recusou (baixa confianca)
python -P src/rocole_hough_treinar.py extrair   # circulos do Hough e atributos (CPU, ~2,5 min)
python -P src/rocole_hough_treinar.py cv        # 10 variantes x 3 dobras (CPU, ~20 min); escolhe pela regra
python -P src/rocole_hough_treinar.py final     # ajusta em treino + validacao; grava models/rocole_hough.joblib
python -P src/rocole_hough_treinar.py fusao     # fusao com a ResNet-50, fora da dobra (usa models/cv/)
python -P src/rocole_hough_avaliar.py           # modelo de Hough no teste, uma vez
python -P src/rocole_hough_avaliar.py --fusao   # fusao no teste, uma vez
python -P src/rocole_hough_avaliar.py --folhas  # F1 de ferrugem pela folha, das predicoes ja gravadas
python -P src/rocole_hough_avaliar.py --conferencia-externa  # fusao do RoCoLe em folhas do BRACOL (~15 min)
python -P src/rocole_hough_treinar.py cv-repetida  # auditoria: 5 repeticoes x 3 dobras (CPU, ~25 min; retoma se parar)
python -P src/rocole_hough_recusa.py cv         # opcao de recusa: escolhe a taxa fora da dobra
python -P src/rocole_hough_recusa.py teste      # recusa no teste, uma vez (das predicoes ja gravadas)
# o mesmo modelo treinado e testado no BRACOL (precisa de data/bracol_raw e de outputs/hough_bracol/metadados.csv)
python -P src/rocole_hough_treinar.py mascaras --bracol   # SAM nas 1.401 folhas (GPU, ~15 min)
python -P src/rocole_hough_treinar.py extrair --bracol
python -P src/rocole_hough_treinar.py cv --bracol
python -P src/rocole_hough_treinar.py final --bracol      # grava models/bracol_hough.joblib
python -P src/rocole_hough_avaliar.py --bracol            # teste do BRACOL, uma vez
python -P app/app.py                            # aplicativo (modelo do BRACOL) em http://127.0.0.1:5000
python -P app/app.py --modelo rocole            # aplicativo com o detector do RoCoLe (fusao e "Incerto")
python -m unittest discover -s tests     # testes
```

Treinar de novo na GPU dá números próximos, mas não idênticos.

Se os dados estiverem em outra pasta (por exemplo, num worktree do git, que não tem `data/`), aponte a variável de
ambiente `ROCOLE_DADOS` para ela: `export ROCOLE_DADOS=/caminho/para/data` (no PowerShell:
`$env:ROCOLE_DADOS = "C:\caminho\para\data"`).

## Referências

- Parraga-Alava, J. et al. *RoCoLe: A robusta coffee leaf images dataset for evaluation of machine learning based
  methods in plant diseases recognition*. Data in Brief 25, 104414, 2019. Dados: Mendeley Data `c5yvn32dzg`,
  CC BY 4.0.
- Rodriguez-Gallo, Escobar-Benitez e Rodriguez-Lainez. *Robust Coffee Rust Detection Using UAV-Based Aerial RGB
  Imagery*. AgriEngineering 5, 2023.
- Kirillov, A. et al. *Segment Anything*. ICCV, 2023 (modelo `facebook/sam-vit-base`).
- He, K. et al. *Deep Residual Learning for Image Recognition*. CVPR, 2016 (ResNet-50).
- Duda, R. O. e Hart, P. E. *Use of the Hough Transformation to Detect Lines and Curves in Pictures*.
  Communications of the ACM 15(1), 1972.
- Yuen, H. K. et al. *Comparative study of Hough Transform methods for circle finding*. Image and Vision Computing
  8(1), 1990 (método em duas etapas usado pelo `HOUGH_GRADIENT` do OpenCV).
- Ojala, T., Pietikäinen, M. e Mäenpää, T. *Multiresolution Gray-Scale and Rotation Invariant Texture Classification
  with Local Binary Patterns*. IEEE TPAMI 24(7), 2002 (LBP).
- Haralick, R. M., Shanmugam, K. e Dinstein, I. *Textural Features for Image Classification*. IEEE Transactions on
  Systems, Man, and Cybernetics 3(6), 1973 (GLCM).
- Geurts, P., Ernst, D. e Wehenkel, L. *Extremely randomized trees*. Machine Learning 63, 2006.
- Krohling, R. A., Esgario, J. e Ventura, J. A. *BRACOL: A Brazilian Arabica Coffee Leaf images dataset*. Mendeley
  Data, 2019 (conferência externa).
