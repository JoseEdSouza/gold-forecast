# Baseline

Experimento de baseline com regressão linear sobre lags dos log-retornos do ouro.

## Arquivos

- `baseline_linear_regression.py`: script principal do baseline.

## Entrada

- `data/processed/final_gold_data.csv`

## Saídas

- `data/artifacts/baseline/*.png`
- `data/artifacts/baseline/baseline_lr_metricas.csv`
- `data/artifacts/baseline/baseline_lr_model.pkl`
- `data/artifacts/baseline/baseline_lr_scaler.pkl`

## Objetivo

Fornecer uma referência simples e interpretável para comparar com os modelos mais complexos de `price_cnn` e `volatility`.

## Execução

```bash
python experiments/baseline/baseline_linear_regression.py
```
