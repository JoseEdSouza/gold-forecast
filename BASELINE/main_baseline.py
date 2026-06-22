# %%
from statsmodels.tsa.stattools import adfuller
import numpy as np
from pathlib import Path
import os 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import logging
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf as compute_pacf
import joblib
from tensorflow import keras
from keras.callbacks import EarlyStopping
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import timedelta
import matplotlib.gridspec as gridspec


# %%
logging.basicConfig(level=logging.INFO, 
format="%(asctime)s [%(levelname)s] %(message)s",
datefmt= "%d %H: %M: %S")

log = logging.getLogger(__name__)

#%% hiperparâmetros
SEED = 42
GOLD_DATA_PATH = "final_gold_data.csv"
OUTPUT_DIR = Path("output_analisefinal")
OUTPUT_DIR.mkdir(exist_ok=True)

RAMDOM_SEED= 42
WINDOW_SIZE_MIN = 10  # Valor mínimo/fallback; será ajustado com base em lags significativos da PACF
WINDOW_SIZE_MAX = 50  # Limite máximo para evitar modelo muito pesado
WINDOW_SIZE = WINDOW_SIZE_MIN 

VAL_SPLIT = 0.1
LIMIAR_SIGMA = 2.0
JANELA_VOL = 21
LAGS = 60
HORIZONTE = 5
TRAIN_RATIO = 0.95
# %% configurando reprodutibilidade
log.info("\n config da reprodutibilidade")
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
    log.info("Tensorflo seed configurado: %d", SEED)
except ImportError:
    log.warning("Tensorflow não disponível")
    
# %% carregar os dados
log.info("Carregaod os dados")

""" GOLD_DATA_PATH = Path(
    os.environ.get("GOLD_DATA_PATH", "../data/final_gold_data.csv")
) """

df = pd.read_csv('final_gold_data.csv', sep=";", encoding="utf-8", parse_dates=["timestamp"])
log.info("Shape: %s", df.shape)
log.info("Tipos:\n%s", df.dtypes)
log.info("Estatísticas:\n%s", df.describe())

log.info("Nulos por coluna:\n%s", df.isnull().sum())

# %%
datas = pd.to_datetime(df["timestamp"])
gaps = datas.diff().dropna()
gaps_grandes = gaps[gaps > pd.Timedelta("3 days")]
log.info("Gaps > 3 dias: %d", len(gaps_grandes))

invalidos = df[df["close"] <= 0]
log.info("Linhas com close <= 0: %d", len(invalidos))

#%% visualizacao inicial
plt.figure(figsize=(14, 5))
plt.plot(datas, df["close"], linewidth=0.8)
plt.title("Preço de fechamento do ouro")
plt.xlabel("Data")
plt.ylabel("USD")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_preco_fechamento.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 01_preco_fechamento.png")

# %% log
log.info("transformando em log")

df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

log.info("Série de retornos: primeiras linhas:\n%s",
         df[["timestamp", "close", "log_return"]].head())

# %% teste de estacionariedade (ADF)
log.info("\nTestando estacionariedade")

def testar_estacionariedade(serie, nome):
    resultado = adfuller(serie.dropna())
    p_value = resultado[1]
    estacionaria = p_value < 0.05
    log.info(
        "ADF [%s]: estatística: %.4f | p-value: %.4f | estacionária: %s",
        nome, resultado[0], p_value, estacionaria
    )
    return estacionaria

testar_estacionariedade(df["close"], "preco_fechamento")
testar_estacionariedade(df["log_return"], "log_return")

#%% distribuição dos log returns
log.info("\ndistribuição dos retornos")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df["log_return"], bins=80, density=True, alpha=0.7, color="#378ADD")
mu, sigma = df["log_return"].mean(), df["log_return"].std()
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
axes[0].plot(x, stats.norm.pdf(x, mu, sigma), color="#E24B4A", linewidth=1.5,
             label="Normal teórica")
axes[0].set_title("Distribuição dos retornos")
axes[0].legend()
axes[0].grid(True)

stats.probplot(df["log_return"], plot=axes[1])
axes[1].set_title("Q-Q plot (vs normal)")
axes[1].grid(True)

axes[2].boxplot(df["log_return"], vert=True)
axes[2].set_title("Boxplot dos retornos")
axes[2].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_distribuicao_retornos.png", dpi=150)
plt.close()

curtose = df["log_return"].kurt()
assimetria = df["log_return"].skew()
log.info("Curtose: %.4f | Assimetria: %.4f", curtose, assimetria)


# %% detecção de outliers (eventos extremos)
log.info("\nDetectando eventos extremos")

df["extremo"] = np.abs(df["log_return"]) > LIMIAR_SIGMA * df["log_return"].std()
eventos = df[df["extremo"]][["timestamp", "close", "log_return"]]
log.info("Eventos extremos (> %.1f desvios): %d ocorrências", LIMIAR_SIGMA, len(eventos))
log.info("Primeiros eventos:\n%s", eventos.head(10).to_string())

plt.figure(figsize=(14, 5))
plt.plot(df["timestamp"], df["log_return"], linewidth=0.6, alpha=0.8, label="Log return")
plt.scatter(eventos["timestamp"], eventos["log_return"],
            color="#E24B4A", s=20, zorder=5, label=f"Extremos (>{LIMIAR_SIGMA}σ)")
plt.axhline(0, color="gray", linewidth=0.5)
plt.title("Retornos logarítmicos e eventos extremos")
plt.xlabel("Data")
plt.ylabel("Log return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_eventos_extremos.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 03_eventos_extremos.png")

# %% volatilidade
log.info("\nAnalisando volatilidade")

df["volatilidade"] = df["log_return"].rolling(window=JANELA_VOL).std() * np.sqrt(252)

plt.figure(figsize=(14, 4))
plt.plot(df["timestamp"], df["volatilidade"], linewidth=0.8, color="#F39C12")
plt.title(f"Volatilidade anualizada (janela {JANELA_VOL} dias)")
plt.xlabel("Data")
plt.ylabel("Volatilidade")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_volatilidade.png", dpi=150)
plt.close()

vol_max_idx = df["volatilidade"].idxmax()
vol_max = df.loc[vol_max_idx, ["timestamp", "volatilidade"]]
log.info("Pico de volatilidade: %s", vol_max.to_dict())

# %% decomposição da série
log.info("\n[8] Decompondo série...")

decomp = seasonal_decompose(df["close"].values, model="multiplicative", period=252)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
componentes = ["Observado", "Tendência", "Sazonalidade", "Resíduo"]
dados = [df["close"].values, decomp.trend, decomp.seasonal, decomp.resid]

for ax, comp, dado in zip(axes, componentes, dados):
    ax.plot(dado, linewidth=0.7)
    ax.set_ylabel(comp)
    ax.grid(True)

axes[0].set_title("Decomposição da série (preço fechamento)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_decomposicao.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 05_decomposicao.png")

#%%  acf e pacf
log.info("\nAnalisando ACF e PACF")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
plot_acf(df["log_return"], lags=LAGS, ax=ax1, title="ACF — retorno logarítmico")
plot_pacf(df["log_return"], lags=LAGS, ax=ax2, title="PACF — retorno logarítmico")
ax1.grid(True)
ax2.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_acf_pacf.png", dpi=150)
plt.close()

pacf_vals = compute_pacf(df["log_return"].dropna(), nlags=LAGS)
conf_bound = 1.96 / np.sqrt(len(df["log_return"].dropna()))
lags_sig = [i for i, v in enumerate(pacf_vals) if abs(v) > conf_bound and i > 0]
log.info("Lags significativos na PACF: %s", lags_sig)


if lags_sig:
    window_sugerido = max(lags_sig)
else:
    window_sugerido = WINDOW_SIZE_MIN


WINDOW_SIZE = max(WINDOW_SIZE_MIN, min(window_sugerido, WINDOW_SIZE_MAX))
log.info("Window_size sugerido pela PACF: %d", window_sugerido)
log.info("Window_size final (após limites): %d", WINDOW_SIZE)

# %% correlação 
log.info("\ncorrelação entre variáveis")

df_num = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(9, 7))
sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.4, vmin=-1, vmax=1)
plt.title("Correlação entre variáveis numéricas")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_correlacao.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 07_correlacao.png")

# %% BASELINE: REGRESSÃO LINEAR — imports e setup
log.info("\n=== BASELINE: REGRESSÃO LINEAR ===")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# %% 1. FEATURE ENGINEERING — lags determinados pela PACF
log.info("Construindo %d features de lag (janela PACF)", WINDOW_SIZE)

feature_cols = [f"lag_{i}" for i in range(1, WINDOW_SIZE + 1)]
for i in range(1, WINDOW_SIZE + 1):
    df[f"lag_{i}"] = df["log_return"].shift(i)

# target: log_return do próximo dia (previsão 1-step ahead)
# shift(-1): target[t] = log_return[t+1]
df["target"] = df["log_return"].shift(-1)

# elimina NaN gerados pelos shifts (início e fim da série)
df_model = (
    df[feature_cols + ["target", "timestamp", "close"]]
    .dropna()
    .reset_index(drop=True)
)
log.info("Shape do dataset modelável: %s", df_model.shape)


# %% 2. SPLIT TEMPORAL (sem embaralhamento — essencial para séries temporais)
n = len(df_model)
n_train = int(n * TRAIN_RATIO)

train_df = df_model.iloc[:n_train].copy()
test_df  = df_model.iloc[n_train:].copy()

X_train = train_df[feature_cols].values
y_train = train_df["target"].values
X_test  = test_df[feature_cols].values
y_test  = test_df["target"].values

log.info("Treino: %d amostras | Teste: %d amostras", len(train_df), len(test_df))
log.info("Período treino: %s → %s",
         train_df["timestamp"].iloc[0].date(),
         train_df["timestamp"].iloc[-1].date())
log.info("Período teste:  %s → %s",
         test_df["timestamp"].iloc[0].date(),
         test_df["timestamp"].iloc[-1].date())


# %% 3. NORMALIZAÇÃO (fit apenas no treino → sem data leakage)
scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)


# %% 4. TREINO
model_lr = LinearRegression()
model_lr.fit(X_train_sc, y_train)

log.info("Intercepto: %.6f", model_lr.intercept_)
log.info("Coef. mais relevantes (top 5 por magnitude):\n%s",
         pd.Series(model_lr.coef_, index=feature_cols)
           .abs().nlargest(5).to_string())


# %% 5. BASELINE INGÊNUO (persistência)
# Hipótese: o retorno de amanhã = o retorno de hoje (lag_1)
y_naive = X_test[:, 0]   # lag_1 = retorno do dia anterior


# %% 6. PREDIÇÃO
y_pred_train = model_lr.predict(X_train_sc)
y_pred_test  = model_lr.predict(X_test_sc)


# %% 7. MÉTRICAS
def calcular_metricas(y_true, y_pred, label):
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    mae      = mean_absolute_error(y_true, y_pred)
    r2       = r2_score(y_true, y_pred)
    dir_acc  = np.mean(np.sign(y_true) == np.sign(y_pred))
    log.info(
        "[%s] RMSE: %.6f | MAE: %.6f | R²: %.4f | Dir.Acc.: %.4f",
        label, rmse, mae, r2, dir_acc
    )
    return {"rmse": rmse, "mae": mae, "r2": r2, "dir_acc": dir_acc}

metricas_treino = calcular_metricas(y_train,  y_pred_train, "LR  TREINO")
metricas_teste  = calcular_metricas(y_test,   y_pred_test,  "LR  TESTE ")
metricas_naive  = calcular_metricas(y_test,   y_naive,      "NAIVE TESTE")

df_metricas = pd.DataFrame(
    [metricas_treino, metricas_teste, metricas_naive],
    index=["lr_train", "lr_test", "naive_test"]
)
df_metricas.to_csv(OUTPUT_DIR / "baseline_lr_metricas.csv")
log.info("Métricas salvas.")


# %% 8. ANÁLISE DE RESÍDUOS — Ljung-Box
residuos = y_test - y_pred_test

lb = acorr_ljungbox(residuos, lags=[10, 20], return_df=True)
log.info("Ljung-Box nos resíduos (autocorrelação residual):\n%s", lb.to_string())
# p-value alto → resíduos sem autocorrelação significativa (bom sinal)


# %% 9a. VISUALIZAÇÃO — Real vs Predito + Resíduos
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(test_df["timestamp"], y_test,       lw=0.7, label="Real",   alpha=0.85)
axes[0].plot(test_df["timestamp"], y_pred_test,  lw=0.7, label="LR",     alpha=0.85)
axes[0].plot(test_df["timestamp"], y_naive,      lw=0.5, label="Naive",  alpha=0.5,
             linestyle="--", color="gray")
axes[0].set_title("Baseline LR — Log Return: Real vs Predito (teste)")
axes[0].set_ylabel("Log return")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(test_df["timestamp"], residuos, lw=0.6, color="#E24B4A", alpha=0.8)
axes[1].axhline(0, color="gray", lw=0.8)
axes[1].set_title("Resíduos (Real − Predito)")
axes[1].set_ylabel("Resíduo")
axes[1].set_xlabel("Data")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_baseline_lr_predicoes.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 08_baseline_lr_predicoes.png")


# %% 9b. VISUALIZAÇÃO — Reconstrução do preço acumulado
preco_inicio = test_df["close"].iloc[0]
preco_real   = preco_inicio * np.exp(np.cumsum(y_test))
preco_pred   = preco_inicio * np.exp(np.cumsum(y_pred_test))
preco_naive  = preco_inicio * np.exp(np.cumsum(y_naive))

plt.figure(figsize=(14, 5))
plt.plot(test_df["timestamp"], preco_real,  lw=1.0, label="Preço real")
plt.plot(test_df["timestamp"], preco_pred,  lw=0.8, label="LR",    linestyle="--")
plt.plot(test_df["timestamp"], preco_naive, lw=0.6, label="Naive", linestyle=":",
         color="gray", alpha=0.7)
plt.title("Reconstrução do preço via log returns acumulados (teste)")
plt.ylabel("USD")
plt.xlabel("Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_baseline_lr_preco_reconstruido.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 09_baseline_lr_preco_reconstruido.png")


# %% 9c. VISUALIZAÇÃO — Distribuição e Q-Q dos resíduos
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

mu_r, sig_r = residuos.mean(), residuos.std()
xr = np.linspace(mu_r - 4*sig_r, mu_r + 4*sig_r, 200)
axes[0].hist(residuos, bins=60, density=True, alpha=0.7, color="#378ADD")
axes[0].plot(xr, stats.norm.pdf(xr, mu_r, sig_r), color="#E24B4A", lw=1.5,
             label="Normal teórica")
axes[0].set_title("Distribuição dos resíduos")
axes[0].legend()
axes[0].grid(True)

stats.probplot(residuos, plot=axes[1])
axes[1].set_title("Q-Q dos resíduos")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_baseline_lr_residuos.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 10_baseline_lr_residuos.png")


# %% 9d. VISUALIZAÇÃO — Importância dos coeficientes
coefs = pd.Series(model_lr.coef_, index=feature_cols)
plt.figure(figsize=(max(10, WINDOW_SIZE // 2), 4))
coefs.plot(kind="bar", color="#378ADD", alpha=0.8)
plt.axhline(0, color="gray", lw=0.8)
plt.title("Coeficientes da Regressão Linear (features normalizadas)")
plt.xlabel("Lag")
plt.ylabel("Coeficiente")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_baseline_lr_coeficientes.png", dpi=150)
plt.close()
log.info("Gráfico salvo: 11_baseline_lr_coeficientes.png")


# %% 10. PERSISTÊNCIA DO MODELO
joblib.dump(model_lr,  OUTPUT_DIR / "baseline_lr_model.pkl")
joblib.dump(scaler_X,  OUTPUT_DIR / "baseline_lr_scaler.pkl")
log.info("Modelo e scaler salvos em: %s", OUTPUT_DIR)


# %% 11. RESUMO FINAL
log.info("\n%s", "="*55)
log.info("RESUMO DO BASELINE")
log.info("%s", "="*55)
log.info("\n%s", df_metricas.to_string())
log.info("="*55)