# Detector de ferrugem do café por foto da folha

Projeto da disciplina **SIN 323 (Inteligência Artificial, UFV, 2026/2)**: um aplicativo local que recebe a foto de uma
folha de café, procura sinais de **ferrugem** e mostra onde encontrou esses sinais.

> É um **apoio à decisão**. Não substitui o agrônomo.

![Tela de resultado do aplicativo](docs/app/resultado_bracol.png)

**Quer entender o projeto inteiro?** Leia a [explicação em Word](docs/Explicacao_do_Projeto.docx).

## Resultados principais

Todos os números são do **conjunto de teste**, com imagens que os modelos nunca viram. IC 95% = intervalo de confiança
de 95% por bootstrap.

| Pergunta | Modelo | Teste | Resultado | IC 95% |
|---|---|---|---|---|
| Há ferrugem? (folha destacada, fundo claro) | Hough + classificador | BRACOL, 201 folhas | F1 **0,949** | 0,912 a 0,979 |
| Há ferrugem? (foto de campo) | Hough + ResNet-50 | RoCoLe, 208 fotos | F1 **0,879** | 0,774 a 0,934 |
| Há ferrugem? (campo, com opção de recusa) | o mesmo, respondendo a 86,1% das fotos | RoCoLe, 179 fotos aceitas | F1 **0,935** | 0,845 a 0,980 |
| Saudável, ácaro-vermelho ou ferrugem? | ResNet-50, 512×288 | RoCoLe, 208 fotos | F1 macro **0,753** | 0,650 a 0,829 |

Os números de conjuntos diferentes não se comparam diretamente: as fotos, as espécies e as classes mudam.
Detalhes e arquivo de origem de cada número: [docs/RESULTADOS.md](docs/RESULTADOS.md).

## Como rodar o aplicativo

Requer o [Anaconda](https://www.anaconda.com/download). O ambiente usa Python 3.13. O uso de GPU é recomendado. No **Anaconda Prompt**, na raiz do projeto:

```bash
conda create -n cafe-ferrugem python=3.13 pip
conda activate cafe-ferrugem
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
python -P app/app.py
```

Nas próximas vezes, basta rodar `conda activate cafe-ferrugem` e `python -P app/app.py`.

Depois abra <http://127.0.0.1:5000>. O modelo padrão é o do BRACOL. Para fotos de campo, use o modelo do RoCoLe, que
também pode responder "Incerto":

```bash
python -P app/app.py --modelo rocole
```

Na primeira análise, o segmentador SAM (cerca de 375 MB) é baixado e os modelos são carregados.

Para rodar os testes, com o ambiente ativado e na raiz do projeto:

```bash
set PYTHONPATH=src
python -P -m unittest discover -s tests
```

No Linux ou no macOS, troque a primeira linha por `export PYTHONPATH=src`.

## Organização da pasta

```text
.
├── README.md
├── requirements.txt
├── app/                     aplicativo local (Flask)
├── src/                     código do detector de ferrugem e da ResNet-50
├── tests/                   testes automáticos
├── models/                  modelos finais
├── outputs/                 resultados, separados por modelo
├── docs/
│   ├── Explicacao_do_Projeto.docx
│   ├── RESULTADOS.md        números finais e arquivo de origem
│   ├── SELECAO.md           o que entrou nesta pasta, e por quê
│   ├── RELATORIO_ROCOLE.md  relatório completo do detector e do aplicativo
│   ├── RELATORIO_BRACOL.md  relatório do modelo de 4 doenças
│   ├── MANIFESTO.csv        cada arquivo, sua origem e seu SHA-256
│   ├── app/                 telas do aplicativo
│   └── specs/               desenho dos métodos, escrito antes dos experimentos
└── resultados_4_doencas/    pesos e resultados do modelo de 4 doenças
```

## Pesos dos modelos

| Arquivo | Modelo | Uso |
|---|---|---|
| `models/bracol_hough.joblib` | detector de ferrugem treinado no BRACOL | modelo padrão do aplicativo |
| `models/rocole_hough.joblib` | detector de ferrugem treinado no RoCoLe, com fusão e recusa | `--modelo rocole` |
| `models/rocole_resnet50.pt` | ResNet-50 treinada no RoCoLe | parte da fusão de `--modelo rocole` |
| `resultados_4_doencas/models/melhor_modelo.pt` | ConvNeXt-Tiny de 4 doenças e severidade | não é usado pelo aplicativo |

## Dados

As imagens não fazem parte desta pasta. Para treinar ou avaliar de novo, baixe os dois conjuntos (licença CC BY 4.0),
coloque-os em `data/` e siga o [relatório completo](docs/RELATORIO_ROCOLE.md#como-reproduzir):

- **BRACOL:** <https://data.mendeley.com/datasets/yy2k5y8mxg>
- **RoCoLe:** Mendeley Data `c5yvn32dzg`

## Limitações

- Cada modelo foi medido só no tipo de foto em que foi treinado. Não há um número que valha para qualquer folha de café.
- Em fotos de campo, o detector confunde ácaro-vermelho com ferrugem.
- Os conjuntos de teste são pequenos, por isso os intervalos de confiança são largos.
