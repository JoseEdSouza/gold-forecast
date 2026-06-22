# 03 — GRU Multi-horizonte

Previsão multi-horizonte de log-retornos do ouro com rede recorrente GRU (Gated Recurrent Unit).

## Objetivo

Explorar a capacidade das redes recorrentes de capturar dependências temporais de longo prazo
na série do ouro. O modelo aprende simultaneamente os três horizontes de previsão com cabeças
de saída independentes e perda de Huber para robustez a outliers.

## Arquivos

| Arquivo | Descrição |
| --- | --- |
| `gold_price_gru.py` | Script principal: construção, treino e avaliação do modelo |
| `gold_price_gru.ipynb` | Notebook equivalente com visualizações inline |

## Entrada

- `data/processed/gold_features.csv` — features técnicas e macro derivadas da série do ouro

## Arquitetura

```text
Input (LOOKBACK=60, n_features)
  └─ GRU(128, return_sequences=True)
       └─ GRU(64)
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

## Saídas (`outputs/03_gru/`)

- `gold_price_gru.keras` — modelo treinado serializado
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
python experiments/03_gru/gold_price_gru.py
```
