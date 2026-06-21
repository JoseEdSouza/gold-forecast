# Gold Forecast

Repositório reorganizado para separar dados, experimentos, artefatos e modelos.

## Estrutura

- `data/processed`: datasets tabulares usados pelos experimentos.
- `data/external`: documentação de datasets externos esperados, mas não versionados.
- `data/artifacts`: saídas geradas por treino, avaliação e análise.
- `experiments/price_cnn`: linha de experimentos de previsão de preço por log-retorno com CNN 1D.
- `experiments/volatility`: linha de experimentos de previsão de volatilidade com GRU e features macro.
- `scripts`: utilitários de download e geração de features.
- `models`: modelos treinados salvos.

## Entradas principais

- `data/processed/final_gold_data.csv`
- `data/processed/final_silver_data.csv`
- `data/processed/final_platinum_data.csv`
- `data/processed/final_palladium_data.csv`
- `data/processed/gold_features.csv`

## Experimentos

- Preço: [experiments/price_cnn/README.md](/home/jose_edsouza/Documentos/Faculdade/Mestrado/ML/Trabalhos/repo/gold-forecast/experiments/price_cnn/README.md)
- Volatilidade: [experiments/volatility/README.md](/home/jose_edsouza/Documentos/Faculdade/Mestrado/ML/Trabalhos/repo/gold-forecast/experiments/volatility/README.md)
- Dados: [data/README.md](/home/jose_edsouza/Documentos/Faculdade/Mestrado/ML/Trabalhos/repo/gold-forecast/data/README.md)
