# Experimento 02: XGBoost com Otimizacao Bayesiana

## Objetivo

Este experimento introduz um modelo tabular nao linear para a trilha principal multi-horizonte. A ideia e explorar interacoes entre features tecnicas e macroeconomicas sem assumir dinamica puramente linear.

## Hipotese

Um conjunto rico de features em janelas temporais pode ser melhor explorado por gradient boosting do que por um baseline linear, principalmente em relacoes nao lineares e combinacoes locais de sinais.

## Arquivos principais

| Arquivo | Papel |
| --- | --- |
| `experiments/02_xgboost/gold_price_xgboost_cv_bayes.py` | Script principal com treino, tuning e avaliacao |
| `experiments/02_xgboost/gold_price_xgboost_cv_bayes.ipynb` | Notebook equivalente |
| `experiments/02_xgboost/gold_price_xgboost.ipynb` | Exploracao inicial do XGBoost |

## Entrada

- `data/processed/gold_features.csv`

## Contrato do experimento

- Horizontes: `h=5`, `h=15`, `h=30`
- Lookback: `60`
- Split principal: `70% / 15% / 15%` em ordem temporal
- Gap anti-vazamento: `max(HORIZONS)`
- Validacao cruzada: `TimeSeriesSplit` com `5` folds
- Tuning: `BayesSearchCV` com `16` iteracoes por horizonte
- Um modelo independente por horizonte

## Pipeline

1. Ler o dataset enriquecido.
2. Selecionar features numericas validas.
3. Construir janelas temporais de `60` observacoes.
4. Achatar cada janela em um vetor tabular.
5. Fazer tuning por horizonte com validacao cruzada temporal.
6. Treinar o melhor estimador por horizonte.
7. Avaliar no conjunto de teste e salvar importancias, metricas e graficos.

## Execucao

```bash
python experiments/02_xgboost/gold_price_xgboost_cv_bayes.py
```

## Saidas em `outputs/02_xgboost/`

- `best_params.json` e `best_params.csv`
- `cv_top10_per_horizon.csv`
- `metrics_test_h5.csv`, `metrics_test_h15.csv`, `metrics_test_h30.csv`
- `feature_importance_top30_h5.csv`, `feature_importance_top30_h15.csv`, `feature_importance_top30_h30.csv`
- `feature_importance_top30.png`
- `predicted_vs_actual_test.png`
- `yearly_h30.csv` e `yearly_h30.png`
- `residuals_h30.png`, `scatter_test.png`, `returns_test.png`, `split_regions.png`, `price_series.png`

## Como interpretar

- Compare `MAE/naive` e `DirAcc%` por horizonte.
- Use a importancia de features para identificar quais sinais sobrevivem ao tuning.
- O resultado por ano em `h=30` ajuda a avaliar dependencia de regime.

## Quando preferir este experimento

- Quando voce quer um modelo mais rapido que redes neurais profundas.
- Quando interpretabilidade de features ainda importa.
- Quando faz sentido explorar tuning cuidadoso por horizonte.

## Limitacoes

- O modelo perde a estrutura temporal explicita ao achatar janelas.
- Cada horizonte e treinado isoladamente, sem compartilhamento de representacao.
- O custo do tuning cresce com o espaco de busca e com o numero de folds.
