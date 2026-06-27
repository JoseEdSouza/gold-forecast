# Experimento 05: LSTM Multi-horizonte

## Objetivo

Este experimento replica a trilha recorrente multi-horizonte com LSTM para comparar explicitamente o ganho ou custo de uma memoria mais estruturada em relacao a GRU.

## Hipotese

Se a serie do ouro exigir retenção de contexto por janelas mais longas ou relacoes temporais mais complexas, a LSTM pode capturar isso melhor que a GRU, mesmo com custo computacional adicional.

## Arquivos principais

| Arquivo | Papel |
| --- | --- |
| `experiments/05_lstm/gold_price_lstm.py` | Script principal do modelo LSTM |
| `experiments/05_lstm/gold_price_lstm.ipynb` | Notebook principal |
| `experiments/05_lstm/gold_price_lstm_cv.ipynb` | Notebook complementar |

## Entrada

- `data/processed/gold_features.csv`

## Contrato do experimento

- Horizontes: `h=5`, `h=15`, `h=30`
- Lookback: `60`
- Gap: `30`
- Batch size: `64`
- Maximo de epocas: `120`
- Loss: Huber por horizonte
- Arquitetura recorrente com duas camadas LSTM e regularizacao

## Pipeline

1. Ler o dataset enriquecido.
2. Selecionar features numericas validas.
3. Construir janelas temporais com split sem vazamento.
4. Escalonar treino, validacao e teste a partir do treino.
5. Treinar a LSTM com `EarlyStopping` e ajuste de learning rate.
6. Reverter as predicoes para a escala original de log-retorno.
7. Salvar metricas, modelo e visualizacoes padronizadas.

## Arquitetura resumida

```text
Input (60, n_features)
  -> LSTM(64, return_sequences=True)
  -> LayerNormalization
  -> SpatialDropout1D(0.15)
  -> LSTM(64)
  -> LayerNormalization
  -> Dense(64, relu)
  -> Dropout(0.25)
  -> Dense(1) para h5, h15 e h30
```

## Execucao

```bash
python experiments/05_lstm/gold_price_lstm.py
```

## Saidas em `outputs/05_lstm/`

- `gold_price_lstm.keras`
- `metrics.csv`
- `learning_curves.png`
- `predicted_vs_actual_test.png`
- `returns_test.png`
- `scatter_test.png`
- `residuals_h30.png`
- `yearly_h30.csv` e `yearly_h30.png`
- `split_regions.png`, `price_series.png`
- Artefatos adicionais de CV e tuning ja presentes na pasta

## Como interpretar

- Compare diretamente com `03_gru` para isolar o efeito da celula LSTM.
- Se o ganho em erro e estabilidade nao compensar o custo extra, a GRU pode continuar sendo a escolha mais pragmatica.
- Observe se a LSTM melhora principalmente nos horizontes mais longos ou em anos especificos.

## Quando preferir este experimento

- Quando voce quer testar memoria temporal mais estruturada.
- Quando a GRU parecer limitada em janelas mais longas.
- Quando comparacao arquitetural entre recorrentes for uma pergunta central da pesquisa.

## Limitacoes

- Custo de treino maior que a GRU.
- Maior numero de parametros e maior sensibilidade a regularizacao.
- Assim como a GRU, continua dependente de um split principal unico para a avaliacao base.
