# Experimento 01: Baseline com Regressao Linear

## Objetivo

Este experimento estabelece o piso de desempenho do projeto com um modelo simples, interpretavel e barato de executar. Ele funciona como referencia para julgar se os modelos mais complexos realmente agregam valor.

## Hipotese

Os lags recentes dos log-retornos do ouro podem capturar parte suficiente da dinamica de curto prazo para produzir um baseline util, mesmo sem features tecnicas mais ricas ou arquiteturas profundas.

## Arquivos principais

| Arquivo | Papel |
| --- | --- |
| `experiments/01_baseline/baseline_linear_regression.py` | Script principal do baseline |
| `experiments/01_baseline/gold_price_baseline_lr.ipynb` | Notebook com a mesma trilha analitica |

## Entrada

- `data/processed/final_gold_data.csv`

## Contrato do experimento

- Target: log-retorno do proximo dia util (`h=1`)
- Features: lags `1..WINDOW_SIZE` do log-retorno
- Janela: escolhida via PACF, limitada entre `10` e `50`
- Split: temporal `95% / 5%`, sem embaralhamento
- Normalizacao: ajustada apenas no treino

## Pipeline

1. Ler a serie historica do ouro.
2. Calcular log-retornos.
3. Escolher a janela de lags via PACF.
4. Construir matriz supervisionada com lags.
5. Treinar regressao linear.
6. Comparar com um baseline ingenuo de persistencia.
7. Salvar metricas, graficos e artefatos serializados.

## Execucao

```bash
python experiments/01_baseline/baseline_linear_regression.py
```

## Saidas em `outputs/01_baseline/`

- `baseline_lr_metricas.csv`
- `baseline_lr_model.pkl`
- `baseline_lr_scaler.pkl`
- Graficos de preco, distribuicao, eventos extremos, volatilidade, decomposicao, ACF/PACF, correlacao, predicoes, residuos e coeficientes

## Como interpretar

- Use este experimento para verificar se modelos mais sofisticados batem o baseline de forma consistente.
- Compare erro absoluto, RMSE, `R²` e acuracia direcional.
- Se um modelo complexo nao supera de forma clara este baseline, ele provavelmente nao justifica o custo adicional.

## Limitacoes

- Nao usa os horizontes principais `h=5,15,30`.
- Nao usa o dataset enriquecido `gold_features.csv`.
- Serve como referencia, nao como fluxo principal de comparacao multi-horizonte.
