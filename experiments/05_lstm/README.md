# 05 — LSTM Multi-horizonte

Previsão multi-horizonte de log-retornos do ouro com rede recorrente LSTM (Long Short-Term Memory).

## Objetivo

Explorar a capacidade das redes LSTM de capturar dependências temporais de longo prazo
na série do ouro, comparando com o GRU (experimento 03). O LSTM adiciona uma célula de
memória explícita (cell state) que pode reter informação por janelas mais longas,
ao custo de maior número de parâmetros.

## Arquivos

| Arquivo | Descrição |
| --- | --- |
| `gold_price_lstm.py` | Script principal: construção, treino e avaliação do modelo |

## Entrada

- `data/processed/gold_features.csv` — features técnicas e macro derivadas da série do ouro

## Arquitetura

```text
Input (LOOKBACK=60, n_features)
  └─ LSTM(64, return_sequences=True)
       └─ LayerNormalization
            └─ SpatialDropout1D(0.15)
                 └─ LSTM(64)
                      └─ LayerNormalization
                           └─ Dense(64, relu)
                                └─ Dropout(0.25)
                                     ├─ Dense(1)  → h5
                                     ├─ Dense(1)  → h15
                                     └─ Dense(1)  → h30
```

## Configuração principal

| Parâmetro | Valor |
| --- | --- |
| Lookback (janela) | 60 dias |
| Horizontes | h=5, h=15, h=30 dias úteis |
| Épocas | 120 |
| Batch size | 64 |
| Loss | Huber (delta=1.0) por horizonte |
| Target | log-retorno acumulado em h dias |

## Abordagem

- Split temporal com gap de `max(HORIZONS)=30` dias entre treino e teste para evitar vazamento
- Normalização com `StandardScaler` ajustado apenas no treino (via `src/data.py`)
- `EarlyStopping` na validação para evitar overfitting
- Métricas por ano calculadas sobre o conjunto de teste (via `src/metrics.py`)

## Comparação com GRU (03)

A única diferença de arquitetura é a substituição de `GRU` por `LSTM`. O LSTM tem
~33% mais parâmetros por camada devido ao gate de célula adicional. Espera-se desempenho
similar ao GRU; diferenças indicam que a série beneficia (ou não) de memória de longo prazo
mais estruturada.

## Saídas (`outputs/05_lstm/`)

- `gold_price_lstm.keras` — modelo treinado serializado
- `learning_curves.png` — loss de treino e validação por época
- `predicted_vs_actual_test.png` — real vs predito nos três horizontes
- `price_series.png` — série histórica usada
- `split_regions.png` — visualização das regiões de treino/val/teste
- `scatter_test.png` — dispersão real vs predito
- `residuals_h30.png` — resíduos para h=30
- `yearly_h30.{csv,png}` — métricas anuais para h=30
- `metrics.csv` — RMSE, MAE, R², acurácia direcional nos três horizontes

## Execução

```bash
python experiments/05_lstm/gold_price_lstm.py
```
