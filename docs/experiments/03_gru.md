# Experimento 03: GRU Multi-horizonte

## Objetivo

Este experimento move a trilha principal para uma arquitetura recorrente multi-saida. O foco e aprender dependencias temporais diretamente das sequencias de features, sem achatar a janela.

## Hipotese

Uma GRU consegue capturar melhor a ordem temporal e dependencias de medio prazo do que o modelo tabular, compartilhando representacao entre os horizontes `h=5`, `h=15` e `h=30`.

## Arquivos principais

| Arquivo | Papel |
| --- | --- |
| `experiments/03_gru/gold_price_gru.py` | Script principal do modelo GRU |
| `experiments/03_gru/gold_price_gru.ipynb` | Notebook explicativo |
| `experiments/03_gru/gold_price_gru_cv.ipynb` | Notebook adicional de validacao |

## Entrada

- `data/processed/gold_features.csv`

## Contrato do experimento

- Horizontes: `h=5`, `h=15`, `h=30`
- Lookback: `60`
- Gap: `30`
- Batch size: `64`
- Maximo de epocas: `120`
- Loss: Huber por horizonte
- Saida: uma cabeca `Dense(1)` para cada horizonte

## Pipeline

1. Ler o dataset enriquecido.
2. Selecionar features validas e targets `y_5`, `y_15`, `y_30`.
3. Criar janelas temporais com split treino/validacao/teste sem vazamento.
4. Escalonar `X` e `Y` apenas a partir do treino.
5. Treinar a GRU com `EarlyStopping` e `ReduceLROnPlateau`.
6. Reverter a escala das predicoes.
7. Salvar metricas e visualizacoes padronizadas.

## Arquitetura resumida

```text
Input (60, n_features)
  -> GRU(64, return_sequences=True)
  -> LayerNormalization
  -> SpatialDropout1D(0.15)
  -> GRU(64)
  -> LayerNormalization
  -> Dense(64, relu)
  -> Dropout(0.25)
  -> Dense(1) para h5, h15 e h30
```

## Execucao

```bash
python experiments/03_gru/gold_price_gru.py
```

## Saidas em `outputs/03_gru/`

- `gold_price_gru.keras`
- `metrics.csv`
- `learning_curves.png`
- `predicted_vs_actual_test.png`
- `returns_test.png`
- `scatter_test.png`
- `residuals_h30.png`
- `yearly_h30.csv` e `yearly_h30.png`
- `split_regions.png`, `price_series.png`
- Artefatos adicionais de CV ja presentes na pasta

## Como interpretar

- Verifique se a perda converge sem divergir fortemente entre treino e validacao.
- Compare o desempenho por horizonte, especialmente `h=30`, onde a dificuldade costuma ser maior.
- Use `yearly_h30` para avaliar robustez em diferentes regimes de mercado.

## Quando preferir este experimento

- Quando voce quer preservar estrutura temporal explicita.
- Quando faz sentido compartilhar representacao entre horizontes.
- Quando deseja um meio-termo entre custo computacional e expressividade.

## Limitacoes

- O treino e mais caro que XGBoost.
- O comportamento pode ser sensivel a epocas, seed e regularizacao.
- A avaliacao ainda e baseada em um split unico, nao em walk-forward completo.
