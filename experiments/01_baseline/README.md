# 01 — Baseline: Regressão Linear

Modelo de referência simples e interpretável para comparar com os experimentos subsequentes.

## Objetivo

Estabelecer um piso de desempenho usando regressão linear sobre lags dos log-retornos do ouro.
O tamanho da janela de lags é determinado automaticamente pela PACF (máximo lag significativo,
limitado entre 10 e 50). Inclui também um benchmark de persistência ingênua (lag-1).

## Arquivos

| Arquivo | Descrição |
| --- | --- |
| `baseline_linear_regression.py` | Script principal do baseline (EDA + treino + avaliação) |

## Entrada

- `data/processed/final_gold_data.csv`

## Configuração principal

| Parâmetro | Valor |
| --- | --- |
| Target | log-retorno do dia seguinte (h=1) |
| Features | lags 1..`WINDOW_SIZE` do log-retorno |
| Split treino/teste | 95% / 5% (temporal, sem embaralhamento) |
| Normalização | `StandardScaler` (fit só no treino) |
| Window size | automático via PACF (mín 10, máx 50) |

## Saídas (`outputs/01_baseline/`)

- `01_preco_fechamento.png` — série histórica do preço de fechamento
- `02_distribuicao_retornos.png` — histograma, Q-Q plot e boxplot dos log-retornos
- `03_eventos_extremos.png` — retornos com outliers marcados (> 2σ)
- `04_volatilidade.png` — volatilidade anualizada com janela 21 dias
- `05_decomposicao.png` — decomposição multiplicativa da série (tendência, sazonalidade, resíduo)
- `06_acf_pacf.png` — ACF e PACF dos log-retornos
- `07_correlacao.png` — heatmap de correlação entre variáveis numéricas
- `08_baseline_lr_predicoes.png` — real vs predito (LR e naive) no conjunto de teste
- `09_baseline_lr_preco_reconstruido.png` — preço reconstruído via acumulação dos log-retornos
- `10_baseline_lr_residuos.png` — distribuição e Q-Q dos resíduos
- `11_baseline_lr_coeficientes.png` — magnitude dos coeficientes por lag
- `baseline_lr_metricas.csv` — RMSE, MAE, R², acurácia direcional (treino, teste, naive)
- `baseline_lr_model.pkl` — modelo serializado
- `baseline_lr_scaler.pkl` — scaler serializado

## Execução

```bash
python experiments/01_baseline/baseline_linear_regression.py
```
