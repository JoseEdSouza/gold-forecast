# 00 — EDA e Volatilidade

Exploração inicial dos dados e experimentos de previsão de volatilidade do ouro com GRU e features macroeconômicas.

## Objetivo

Entender a estrutura da série histórica do ouro antes de construir modelos preditivos:
estacionariedade, distribuição dos retornos, autocorrelação, sazonalidade e relação com
outros metais e variáveis macro. Inclui também um experimento de previsão de volatilidade
como alternativa ao eixo de previsão de preço.

## Arquivos

| Arquivo | Descrição |
| --- | --- |
| `gold_eda.ipynb` | Análise exploratória geral da série de preço |
| `gold_volatility_exploration.ipynb` | Diagnósticos de volatilidade e ensemble exploratório |
| `gold_volatility_feature_selection.ipynb` | Seleção manual de features para o modelo de volatilidade |
| `gold_volatility_gru.ipynb` | Notebook do modelo GRU de volatilidade com visualizações |
| `gold_volatility_gru.py` | Script equivalente ao notebook acima |

## Entradas

- `data/processed/final_gold_data.csv`
- `data/processed/final_silver_data.csv`
- `data/processed/final_platinum_data.csv`
- `data/processed/final_palladium_data.csv`
- Fontes externas — ver `data/external/README.md`

## Tópicos cobertos na EDA

- Teste de estacionariedade (ADF) no preço e nos log-retornos
- Distribuição dos retornos e comparação com a normal (Q-Q plot, curtose)
- Detecção de eventos extremos por limiar de desvio-padrão
- Volatilidade histórica anualizada com janela móvel de 21 dias
- Decomposição STL (tendência, sazonalidade, resíduo)
- ACF e PACF para determinar lags relevantes
- Correlação entre ouro, prata, platina e paládio

## Observação

Os experimentos de volatilidade dependem de datasets externos não versionados.
O script foi ajustado para o novo layout mas não ficará executável sem esses CSVs adicionais.
