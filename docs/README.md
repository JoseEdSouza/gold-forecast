# Documentacao dos Experimentos

Esta pasta consolida a documentacao operacional dos experimentos do repositorio. O foco aqui e facilitar leitura, reproducao local e comparacao entre abordagens, sem substituir o contrato canonico em [`specs/`](../specs/index.md).

## Indice

| Experimento | Documento | Resumo |
| --- | --- | --- |
| `00_eda` | [experiments/00_eda.md](experiments/00_eda.md) | EDA da serie do ouro e trilha exploratoria de volatilidade |
| `01_baseline` | [experiments/01_baseline.md](experiments/01_baseline.md) | Regressao linear de referencia com lags de log-retorno |
| `02_xgboost` | [experiments/02_xgboost.md](experiments/02_xgboost.md) | XGBoost multi-horizonte com tuning bayesiano |
| `03_gru` | [experiments/03_gru.md](experiments/03_gru.md) | Rede GRU multi-saida para `h=5,15,30` |
| `04_cnn` | [experiments/04_cnn.md](experiments/04_cnn.md) | CNN causal pontual e variante walk-forward com quantis |
| `05_lstm` | [experiments/05_lstm.md](experiments/05_lstm.md) | Rede LSTM multi-saida para comparacao com a GRU |

## Como usar esta pasta

- Comece pelo experimento que voce pretende rodar ou revisar.
- Use o documento correspondente para entender entrada, pre-processamento, estrategia de split, comando de execucao e artefatos gerados.
- Para requisitos estaveis do repositorio, consulte [`specs/intent/capabilities/forecasting.md`](../specs/intent/capabilities/forecasting.md) e [`specs/intent/capabilities/evaluation.md`](../specs/intent/capabilities/evaluation.md).

## Convencoes comuns

- O repositorio trabalha principalmente com previsao de log-retornos acumulados do ouro.
- Os horizontes principais sao `h=5`, `h=15` e `h=30` dias uteis.
- As saidas ficam em `outputs/<experimento>/`, com metricas agregadas tambem copiadas para `outputs/metrics/` quando aplicavel.
- Os utilitarios compartilhados de janelamento, split temporal, metricas e plots ficam em `src/`.
