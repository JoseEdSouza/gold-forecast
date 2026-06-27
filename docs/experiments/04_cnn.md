# Experimento 04: CNN Causal e Walk-Forward com Quantis

## Objetivo

Este experimento cobre duas etapas da linha CNN:

1. previsao pontual multi-horizonte com convolucao causal;
2. extensao para validacao walk-forward com previsao intervalar por quantis.

Ele e o experimento mais orientado a incerteza e a avaliacao fora da amostra ao longo do tempo.

## Hipotese

Convolucoes causais podem extrair padroes locais relevantes da serie do ouro sem vazamento de informacao futura. A variante de quantis deve transformar incerteza e erros de cauda em um sinal observavel, em vez de um unico ponto previsto.

## Arquivos principais

| Arquivo | Papel |
| --- | --- |
| `experiments/04_cnn/gold_price_cnn.py` | CNN pontual com split unico |
| `experiments/04_cnn/gold_price_cnn_walkforward_quantiles.py` | Walk-forward com quantis |
| `experiments/04_cnn/gold_price_cnn.ipynb` | Notebook principal |
| `experiments/04_cnn/gold_price_cnn_cv.ipynb` | Notebook de apoio |
| `experiments/04_cnn/gold_price_cnn_notebook_export.py` | Export de notebook |

## Entrada

- `data/processed/final_gold_data.csv`

## Contrato do experimento

- Horizontes: `h=5`, `h=15`, `h=30`
- Lookback: `60`
- Variante pontual: split unico treino/validacao/teste
- Variante intervalar: folds walk-forward com treino expansivo
- Quantis: `q10`, `q50`, `q90`
- Embargo: `GAP + LOOKBACK`

## Pipeline da variante pontual

1. Ler serie do ouro.
2. Construir features tecnicas e targets no proprio script.
3. Gerar janelas causais.
4. Treinar a CNN multi-saida.
5. Avaliar em split unico e salvar artefatos.

## Pipeline da variante walk-forward com quantis

1. Ler a mesma serie base.
2. Reconstruir features e targets.
3. Definir treino inicial e folds anuais aproximados.
4. Para cada fold, escalar apenas com o treino daquele fold.
5. Treinar a CNN com pinball loss.
6. Produzir `q10`, `q50` e `q90` por horizonte.
7. Agregar predicoes de todos os folds e gerar relatorio consolidado.

## Execucao

```bash
python experiments/04_cnn/gold_price_cnn.py
python experiments/04_cnn/gold_price_cnn_walkforward_quantiles.py
python experiments/04_cnn/gold_price_cnn_walkforward_quantiles.py --folds 1-2
python experiments/04_cnn/gold_price_cnn_walkforward_quantiles.py --report-only
```

## Saidas em `outputs/04_cnn/`

- `gold_price_cnn.keras`
- `metrics_test.csv`
- `predicted_vs_actual_test.png`
- `returns_test.png`
- `scatter_test.png`
- `residuals_h30.png`
- `yearly_h30.csv` e `yearly_h30.png`
- `cv_summary.csv`, `cv_metrics_per_fold.csv`, `cv_vs_test_mae.png`, `cv_learning_curves.png`, `cv_metrics_boxplot.png`
- `predicoes_walkforward.csv` e `predicoes_folds_*.csv` para a variante walk-forward

## Como interpretar

- Na variante pontual, use as mesmas metricas dos demais modelos para comparacao direta.
- Na variante intervalar, `q50` funciona como previsao central e a banda `[q10, q90]` representa incerteza.
- Avalie cobertura empirica da banda e largura media para medir calibracao e confianca.
- Os resultados por fold mostram se o desempenho se sustenta em anos diferentes.

## Quando preferir este experimento

- Quando voce quer previsao com noção explicita de incerteza.
- Quando precisa de avaliacao mais realista ao longo do tempo.
- Quando faz sentido priorizar diagnostico de regime e estabilidade fora da amostra.

## Limitacoes

- E o fluxo mais caro e mais complexo de operar.
- A variante intervalar tem mais hiperparametros conceituais: quantis, tamanho de fold, treino inicial e embargo.
- A calibracao dos quantis nao e automaticamente garantida so porque a loss e adequada.
