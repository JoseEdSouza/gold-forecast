# 04 — CNN 1D Causal com Walk-Forward e Quantis

Previsão multi-horizonte de log-retornos do ouro com CNN 1D causal. Evolui do modelo pontual
para previsão intervalar via quantis com validação walk-forward.

## Objetivo

Usar convoluções causais para extrair padrões locais na série temporal do ouro sem
vazamento de dados futuros. A versão walk-forward com quantis produz intervalos de confiança
(p10, p50, p90) para cada horizonte, tornando a previsão mais útil na prática.

## Arquivos

| Arquivo | Descrição |
| --- | --- |
| `gold_price_cnn.py` | Modelo CNN 1D pontual, split único treino/val/teste |
| `gold_price_cnn.ipynb` | Notebook equivalente com explicações e visualizações |
| `gold_price_cnn_walkforward_quantiles.py` | Evolução com walk-forward e previsão por quantis |
| `gold_price_cnn_notebook_export.py` | Exportação em `.py` do notebook explicativo |
| `gold_price_baseline_lr.ipynb` | Notebook de regressão linear (referência para comparação) |

## Entrada

- `data/processed/final_gold_data.csv`

## Arquitetura — CNN pontual (`gold_price_cnn.py`)

```text
Input (LOOKBACK=60, n_features)
  └─ Conv1D(64, kernel=3, causal padding) + BatchNorm + ReLU
       └─ Conv1D(128, kernel=3, causal padding) + BatchNorm + ReLU
            └─ Conv1D(64, kernel=3, causal padding) + BatchNorm + ReLU
                 └─ GlobalAveragePooling1D
                      ├─ Dense(1)  → h5
                      ├─ Dense(1)  → h15
                      └─ Dense(1)  → h30
```

## Arquitetura — Walk-forward com quantis (`gold_price_cnn_walkforward_quantiles.py`)

```text
Input (LOOKBACK=60, n_features)
  └─ [mesma CNN causal acima]
       ├─ Dense(3)  → h5  [p10, p50, p90]
       ├─ Dense(3)  → h15 [p10, p50, p90]
       └─ Dense(3)  → h30 [p10, p50, p90]
Loss: Pinball loss por quantil e horizonte
```

## Configuração principal

| Parâmetro | CNN pontual | Walk-forward + quantis |
| --- | --- | --- |
| Lookback | 60 dias | 60 dias |
| Horizontes | h=5, h=15, h=30 | h=5, h=15, h=30 |
| Quantis | — | p10, p50, p90 |
| Fold de teste | split único | 252 dias (~1 ano) por fold |
| Treino inicial | 95% | 55% da série |
| Loss | MSE por horizonte | Pinball loss por quantil e horizonte |

## Abordagem walk-forward

- Janela expansiva: a cada fold, o treino cresce acumulando os dados do fold anterior
- Embargo de `GAP + LOOKBACK` amostras entre treino e teste para evitar vazamento
- Opções de execução: `--folds 1-2` (folds específicos) ou `--report-only` (só relatório)

## Saídas (`outputs/04_cnn/`)

- `gold_price_cnn.keras` — modelo CNN pontual treinado
- Predições walk-forward em `outputs/predicoes_walkforward.csv` e `outputs/predicoes_folds_*.csv`
- Predições do modelo pontual em `outputs/predicoes_teste.csv`

## Execução

```bash
# Modelo pontual
python experiments/04_cnn/gold_price_cnn.py

# Walk-forward com quantis — todos os folds
python experiments/04_cnn/gold_price_cnn_walkforward_quantiles.py

# Walk-forward — folds específicos
python experiments/04_cnn/gold_price_cnn_walkforward_quantiles.py --folds 1-2

# Apenas relatório (sem retreino)
python experiments/04_cnn/gold_price_cnn_walkforward_quantiles.py --report-only
```
