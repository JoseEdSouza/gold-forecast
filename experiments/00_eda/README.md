# Volatility

Experimentos de previsão da volatilidade futura do ouro usando séries de mercado, features técnicas e variáveis macro.

## Arquivos

- `gold_volatility_gru.py`: script principal da linha de volatilidade.
- `gold_volatility_gru.ipynb`: notebook base da mesma linha.
- `gold_volatility_exploration.ipynb`: notebook exploratório com diagnósticos extras e ensemble.
- `gold_volatility_feature_selection.ipynb`: versão com seleção manual de features.
- `gold_eda.ipynb`: exploração inicial dos dados.

## Datasets

Arquivos já presentes:
- `data/processed/final_gold_data.csv`
- `data/processed/final_silver_data.csv`
- `data/processed/final_platinum_data.csv`
- `data/processed/final_palladium_data.csv`

Arquivos externos necessários:
- ver [data/external/README.md](/home/jose_edsouza/Documentos/Faculdade/Mestrado/ML/Trabalhos/repo/gold-forecast/data/external/README.md)

## Observação

Essa linha ainda depende de datasets externos não versionados. O script foi ajustado para o novo layout, mas não ficará executável sem esses CSVs adicionais.
