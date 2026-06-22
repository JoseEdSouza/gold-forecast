# Experimentos

Os experimentos seguem uma progressão de complexidade: do modelo mais simples e interpretável
até a previsão intervalar com validação walk-forward.

| # | Pasta | Modelo | Detalhe |
| --- | --- | --- | --- |
| 00 | [00_eda](00_eda/README.md) | — | Exploração inicial e experimentos de volatilidade |
| 01 | [01_baseline](01_baseline/README.md) | Regressão Linear | Lags de log-retorno selecionados via PACF |
| 02 | [02_xgboost](02_xgboost/README.md) | XGBoost | Features técnicas + otimização bayesiana por horizonte |
| 03 | [03_gru](03_gru/README.md) | GRU | Rede recorrente multi-saída com loss de Huber |
| 04 | [04_cnn](04_cnn/README.md) | CNN 1D causal | Previsão pontual e intervalar (quantis) com walk-forward |

Todos os experimentos:

- Preveem **log-retornos acumulados** nos horizontes h=5, h=15 e h=30 dias úteis
- Usam split temporal estrito (sem embaralhamento) com gap para evitar vazamento de dados
- Gravam métricas, plots e modelos em `outputs/<experimento>/`
- Importam utilitários comuns de `src/` (janelamento, métricas, visualização)
