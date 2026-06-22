# 02 — XGBoost com Otimização Bayesiana

Previsão multi-horizonte de log-retornos do ouro com XGBoost e busca de hiperparâmetros via `BayesSearchCV`.

## Objetivo

Substituir a regressão linear por um modelo de gradiente boosting capaz de capturar
relações não-lineares entre features técnicas e os retornos futuros. A busca bayesiana
encontra hiperparâmetros ótimos por horizonte usando validação cruzada temporal.

## Arquivos

| Arquivo | Descrição |
| --- | --- |
| `gold_price_xgboost_cv_bayes.py` | Script principal: treino, tuning e avaliação |
| `gold_price_xgboost_cv_bayes.ipynb` | Notebook equivalente com visualizações inline |
| `gold_price_xgboost.ipynb` | Notebook exploratório inicial do XGBoost |

## Entrada

- `data/processed/gold_features.csv` — features técnicas e macro derivadas da série do ouro

## Configuração principal

| Parâmetro | Valor |
| --- | --- |
| Lookback (janela) | 60 dias |
| Horizontes | h=5, h=15, h=30 dias úteis |
| Split treino/val/teste | 70% / 15% / 15% (temporal) |
| CV splits | 5 folds temporais (`TimeSeriesSplit`) |
| Iterações Bayes | 16 por horizonte |
| Target | log-retorno acumulado em h dias |

## Abordagem

- Um modelo `XGBRegressor` independente por horizonte
- Features: janela deslizante (`LOOKBACK=60`) sobre todas as colunas de `gold_features.csv`
- Tuning via `BayesSearchCV` com `TimeSeriesSplit` para respeitar ordem temporal
- Paralelização dos horizontes via `ProcessPoolExecutor` (CPU) ou loop serial (GPU)

## Saídas (`outputs/02_xgboost/`)

- `best_params.json` / `best_params.csv` — melhores hiperparâmetros por horizonte
- `cv_top10_per_horizon.csv` — top-10 configurações por horizonte na validação cruzada
- `metrics_test_h{5,15,30}.csv` — métricas de teste por horizonte
- `feature_importance_top30_h{5,15,30}.csv` — importância das features (ganho) por horizonte
- `feature_importance_top30.png` — visualização da importância das features
- `predicted_vs_actual_test.png` — real vs predito nos três horizontes
- `yearly_h30.{csv,png}` — desempenho anual para h=30
- `residuals_h30.png`, `scatter_test.png`, `returns_test.png` — diagnósticos do h=30

## Execução

```bash
python experiments/02_xgboost/gold_price_xgboost_cv_bayes.py
```
