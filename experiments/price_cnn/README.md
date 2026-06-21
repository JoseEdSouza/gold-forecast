# Price CNN

Experimentos de previsão do preço do ouro reformulados como previsão de log-retornos acumulados.

## Arquivos

- `gold_price_cnn.py`: script principal com CNN 1D causal multi-horizonte.
- `gold_price_cnn_walkforward_quantiles.py`: evolução com walk-forward e previsão por quantis.
- `gold_price_cnn_notebook_export.py`: exportação em `.py` do notebook explicativo.
- `gold_price_cnn.ipynb`: notebook com explicação, gráficos e métricas.

## Dataset

Entrada principal:
- `data/processed/final_gold_data.csv`

Saídas:
- `data/artifacts/predicoes_teste.csv`
- `data/artifacts/predicoes_walkforward.csv`
- `data/artifacts/predicoes_folds_*.csv`
- `models/gold_price_cnn.keras`

## Execução

```bash
python experiments/price_cnn/gold_price_cnn.py
python experiments/price_cnn/gold_price_cnn_walkforward_quantiles.py --folds 1-2
python experiments/price_cnn/gold_price_cnn_walkforward_quantiles.py --report-only
```
