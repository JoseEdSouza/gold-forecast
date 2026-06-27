# Experimento 00: EDA e Volatilidade

## Objetivo

Esta trilha documenta a fase exploratoria do projeto. Ela serve para entender o comportamento historico do ouro antes do treino de modelos de previsao de preco e, em paralelo, experimentar previsao de volatilidade com features macroeconomicas e recorrencia.

## Quando usar

- Para revisar premissas da serie historica antes de ajustar modelos.
- Para inspecionar estacionariedade, distribuicao dos retornos, autocorrelacao e eventos extremos.
- Para explorar a linha de pesquisa de volatilidade, separada da linha principal de previsao de preco.

## Arquivos principais

| Arquivo | Papel |
| --- | --- |
| `experiments/00_eda/gold_eda.ipynb` | EDA principal da serie do ouro |
| `experiments/00_eda/gold_volatility_exploration.ipynb` | Diagnosticos de volatilidade e exploracoes adicionais |
| `experiments/00_eda/gold_volatility_feature_selection.ipynb` | Selecao manual de features para volatilidade |
| `experiments/00_eda/gold_volatility_gru.ipynb` | Notebook do modelo GRU de volatilidade |
| `experiments/00_eda/gold_volatility_gru.py` | Versao em script do experimento de volatilidade |

## Entradas

- `data/processed/final_gold_data.csv`
- `data/processed/final_silver_data.csv`
- `data/processed/final_platinum_data.csv`
- `data/processed/final_palladium_data.csv`
- Datasets externos adicionais referenciados em `data/external/`

## Perguntas que este experimento responde

- A serie de preco e os log-retornos parecem estacionarios?
- Existem regimes com eventos extremos ou volatilidade elevada?
- Quais lags e correlacoes parecem mais promissores para os modelos supervisionados?
- A previsao de volatilidade parece uma alternativa viavel a previsao direta de preco?

## Analises cobertas

- Teste ADF em preco e log-retornos
- Histograma, boxplot e Q-Q plot dos retornos
- Deteccao de outliers por limiar de desvio-padrao
- Volatilidade anualizada com janela movel
- Decomposicao STL
- ACF e PACF
- Correlacao entre ouro, prata, platina e paladio

## Execucao

Nao existe um unico comando que reproduza toda a trilha `00_eda`. A forma normal de uso e abrir os notebooks e executar por secao. Para a variante GRU de volatilidade:

```bash
python experiments/00_eda/gold_volatility_gru.py
```

## Saidas esperadas

- Graficos exploratorios inline nos notebooks
- Tabelas intermediarias de analise
- Resultados de selecao de features e diagnosticos de volatilidade

## Limitacoes

- A linha de volatilidade depende de dados externos nao versionados no repositorio.
- Essa trilha nao segue o mesmo nivel de padronizacao de artefatos dos experimentos `01` a `05`.
- O foco e exploratorio, nao comparativo.
