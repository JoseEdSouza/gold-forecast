# Gold Forecast

Repositório de experimentos de previsão do preço e volatilidade do ouro usando modelos de machine learning e deep learning.

O problema central é prever log-retornos futuros do ouro em múltiplos horizontes (5, 15, 30 dias úteis), partindo de séries históricas de preço e features técnicas/macroeconômicas.

## Estrutura

```text
gold-forecast/
├── data/
│   ├── raw/            # dados originais baixados (não pré-processados)
│   ├── processed/      # datasets tabulares prontos para uso
│   └── external/       # documentação de fontes externas não versionadas
├── src/                # código compartilhado (data, metrics, plotting)
├── scripts/            # download e geração de features
├── experiments/
│   ├── 00_eda/         # exploração inicial e volatilidade
│   ├── 01_baseline/    # regressão linear (referência)
│   ├── 02_xgboost/     # XGBoost com otimização bayesiana
│   ├── 03_gru/         # GRU multi-horizonte
│   └── 04_cnn/         # CNN 1D causal + walk-forward com quantis
├── outputs/            # artefatos gerados (plots, métricas, modelos)
└── pyproject.toml
```

## Datasets principais

| Arquivo | Descrição |
| --- | --- |
| `data/processed/final_gold_data.csv` | Preço diário do ouro (close, OHLCV) |
| `data/processed/final_silver_data.csv` | Preço diário da prata |
| `data/processed/final_platinum_data.csv` | Preço diário da platina |
| `data/processed/final_palladium_data.csv` | Preço diário do paládio |
| `data/processed/gold_features.csv` | Features técnicas e macro derivadas do ouro |

## Experimentos

| Experimento | Modelo | Entrada | Horizontes |
| --- | --- | --- | --- |
| [00_eda](experiments/00_eda/README.md) | — | `final_gold_data.csv` + macro | exploratório |
| [01_baseline](experiments/01_baseline/README.md) | Regressão Linear | `final_gold_data.csv` | h=1 |
| [02_xgboost](experiments/02_xgboost/README.md) | XGBoost + Bayes CV | `gold_features.csv` | h=5,15,30 |
| [03_gru](experiments/03_gru/README.md) | GRU multi-saída | `gold_features.csv` | h=5,15,30 |
| [04_cnn](experiments/04_cnn/README.md) | CNN 1D + walk-forward | `final_gold_data.csv` | h=5,15,30 |

## Módulo compartilhado (`src/`)

- `src/data.py` — janelamento temporal, splits, normalização
- `src/metrics.py` — RMSE, MAE, R², acurácia direcional, métricas por ano
- `src/plotting.py` — funções padronizadas de visualização para todos os experimentos

## Setup

```bash
uv sync
```

## Execução

```bash
# Gerar features
python scripts/make_dataset.py

# Rodar experimentos individuais
python experiments/01_baseline/baseline_linear_regression.py
python experiments/02_xgboost/gold_price_xgboost_cv_bayes.py
python experiments/03_gru/gold_price_gru.py
python experiments/04_cnn/gold_price_cnn.py
python experiments/04_cnn/gold_price_cnn_walkforward_quantiles.py
```
