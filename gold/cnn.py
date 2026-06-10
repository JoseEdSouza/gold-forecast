#!/usr/bin/env python
# coding: utf-8

# # Célula 1 — Imports

# In[158]:


from pathlib import Path
import random
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
 
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
 
import yfinance as yf
import requests
from arch import arch_model


# # Célula 2 — Constantes 

# In[159]:


START          = "2003-01-02"
END            = "2025-04-14"
TEST_RATIO     = 0.1
SEED           = 42
WINDOW_SIZE    = 20       
VOL_WINDOW     = 5    # era 21
VOL_SHIFT      = -5   # era -21
 
FRED_API_KEY   = "a75bdc9f7da239a5e3d6c8e389345bfb"
DATA_BASE_PATH = Path("./data")
 
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()


# # Célula 3 - Requição para API do FRED

# In[160]:


def get_fred_series(series_id: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()["observations"]
    df = pd.DataFrame(data)[["date", "value"]]
    df.columns = ["timestamp", series_id.lower()]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df[series_id.lower()] = pd.to_numeric(df[series_id.lower()], errors="coerce")
    return df.reset_index(drop=True)
 
dfii10 = get_fred_series("DFII10", START, END, FRED_API_KEY)
dfii10 = dfii10.rename(columns={"dfii10": "inflation"})
print(dfii10.shape)
print(dfii10.head())


# # Células 4, 5, 6 e 7 - Carrega dataframes

# In[161]:


petrodolar = pd.read_csv(
    DATA_BASE_PATH / "petrodollar_5_daily_prices_fx.csv",
    parse_dates=["date"]
)
petrodolar = petrodolar.rename(columns={"date": "timestamp"})
petrodolar = petrodolar[[
    "timestamp",
    "brent_crude_usd",
    "wti_crude_usd",
    "brent_wti_spread",
    "usd_dxy_index",
    "eur_usd",
    "usd_rub",
]]
petrodolar = petrodolar[
    (petrodolar["timestamp"] >= START) &
    (petrodolar["timestamp"] <= END)
].reset_index(drop=True)

print(petrodolar.shape)
print(petrodolar.head())


# In[162]:


opec = pd.read_csv(
    DATA_BASE_PATH / "petrodollar_2_opec_quotas_events.csv",
    parse_dates=["date"]
)
opec = opec.rename(columns={"date": "timestamp"})
opec["timestamp"] = pd.to_datetime(opec["timestamp"])
opec = opec[[
    "timestamp",
    "opec_compliance_pct",
    "oil_market_balance_mbd",
    "us_shale_production_mbd",
]].copy()

# Dataset 4 — Dedolarização
dedolar = pd.read_csv(
    DATA_BASE_PATH / "petrodollar_4_dedollarization.csv",
    parse_dates=["date"]
)
dedolar = dedolar.rename(columns={"date": "timestamp"})
dedolar["timestamp"] = pd.to_datetime(dedolar["timestamp"])
dedolar = dedolar[[
    "timestamp",
    "de_dollarization_pressure_index",
    "brics_gold_reserves_tonnes",
    "usd_global_fx_reserves_pct",
]].copy()

# Dataset 3 — SWF (anual)
swf = pd.read_csv(DATA_BASE_PATH / "petrodollar_3_recycling_swf.csv")
swf = swf[["year", "global_fx_reserves_gold_share_pct"]].copy()

print("OPEC:", opec.shape)
print("Dedolar:", dedolar.shape)
print("SWF:", swf.shape)


# In[163]:


tickers = {
    "dxy":   "DX-Y.NYB",
    "vix":   "^VIX",
    "oil":   "CL=F",
    "tips":  "TIP",
    "sp500": "^GSPC",
    "tnx": "^TNX",
}

dfs = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = name
    dfs[name] = close

externos = pd.concat(dfs.values(), axis=1)
externos.index.name = "timestamp"
externos = externos.reset_index()

print(externos.shape)
print(externos.head())


# In[164]:


gold      = pd.read_csv(DATA_BASE_PATH / "final_gold_data.csv",      sep=";", encoding="utf-8", parse_dates=["timestamp"])
palladium = pd.read_csv(DATA_BASE_PATH / "final_palladium_data.csv", sep=";", encoding="utf-8", parse_dates=["timestamp"])
platinum  = pd.read_csv(DATA_BASE_PATH / "final_platinum_data.csv",  sep=";", encoding="utf-8", parse_dates=["timestamp"])
silver    = pd.read_csv(DATA_BASE_PATH / "final_silver_data.csv",    sep=";", encoding="utf-8", parse_dates=["timestamp"])

cols_to_drop = ["currency", "unit", "headlines"]
for name, df in [("palladium", palladium), ("platinum", platinum), ("silver", silver)]:
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    df.rename(columns={
        "open":   f"{name}_open",
        "high":   f"{name}_high",
        "low":    f"{name}_low",
        "close":  f"{name}_close",
        "volume": f"{name}_volume",
    }, inplace=True)

print("Gold:", gold.shape)


# # Célula 8 - Merges

# In[165]:


for df in [gold, palladium, platinum, silver, externos]:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

gold = gold.merge(palladium,  on="timestamp", how="left")
gold = gold.merge(platinum,   on="timestamp", how="left")
gold = gold.merge(silver,     on="timestamp", how="left")
gold = gold.merge(externos,   on="timestamp", how="left")
gold = gold.merge(dfii10,     on="timestamp", how="left")
gold = gold.merge(petrodolar, on="timestamp", how="left")

# Merge mensal: OPEC e dedolarização (merge por ano+mês)
gold["year"]  = gold["timestamp"].dt.year
gold["month_"] = gold["timestamp"].dt.month

opec["year"]   = opec["timestamp"].dt.year
opec["month_"] = opec["timestamp"].dt.month
opec = opec.drop(columns=["timestamp"])

dedolar["year"]   = dedolar["timestamp"].dt.year
dedolar["month_"] = dedolar["timestamp"].dt.month
dedolar = dedolar.drop(columns=["timestamp"])

gold = gold.merge(opec,    on=["year", "month_"], how="left")
gold = gold.merge(dedolar, on=["year", "month_"], how="left")
gold = gold.merge(swf,     on="year",             how="left")
gold = gold.drop(columns=["year", "month_"])

fill_cols = [
    "dxy", "vix", "oil", "tips", "sp500", "tnx",
    "palladium_close", "palladium_open", "palladium_high", "palladium_low",
    "platinum_close",  "platinum_open",  "platinum_high",  "platinum_low",
    "silver_close",    "silver_open",    "silver_high",    "silver_low",
    "inflation",
    "brent_crude_usd", "wti_crude_usd", "brent_wti_spread", "usd_dxy_index", "eur_usd", "usd_rub",
    "opec_compliance_pct", "oil_market_balance_mbd", "us_shale_production_mbd",
    "de_dollarization_pressure_index", "brics_gold_reserves_tonnes", "usd_global_fx_reserves_pct",
    "global_fx_reserves_gold_share_pct",
]

gold[fill_cols] = gold[fill_cols].ffill().bfill()
gold = gold.dropna()

print("Shape após merge:", gold.shape)
print(gold.isna().sum()[gold.isna().sum() > 0])


# # Célula 9 - Feature Engineering

# In[166]:


gold["day_variation"]  = gold["open"] - gold["close"]
gold["tomorrow_close"] = gold["close"].shift(-1)
gold = gold.dropna(subset=["tomorrow_close"])
 
gold["target_return"] = (gold["tomorrow_close"] - gold["close"]) / gold["close"]
gold["month"]         = gold["timestamp"].dt.month
 
gold["sma_5"]     = gold["close"].rolling(5).mean()
gold["sma_20"]    = gold["close"].rolling(20).mean()
gold["sma_ratio"] = gold["sma_5"] / gold["sma_20"]   # <- adimensional, mantida
 
ema_12 = gold["close"].ewm(span=12).mean()
ema_26 = gold["close"].ewm(span=26).mean()
gold["macd"]        = ema_12 - ema_26
gold["macd_signal"] = gold["macd"].ewm(span=9).mean()
 
rolling_std         = gold["close"].rolling(20).std()
gold["bb_position"] = (gold["close"] - (gold["sma_20"] - 2*rolling_std)) / (4 * rolling_std)  # adimensional [0,1]
 
gold["roc_10"] = gold["close"].pct_change(10)
 
gold["real_rate_proxy"] = gold["tnx"] - gold["inflation"]
gold["petrodolar_brent"] = gold["brent_crude_usd"] / gold["usd_dxy_index"]
gold["brent_roc_20"]     = gold["brent_crude_usd"].pct_change(20)
gold["brent_vol_20"]     = gold["brent_crude_usd"].pct_change().rolling(20).std()
gold["close_return"] = gold["close"].pct_change()
 
delta    = gold["close"].diff()
gain     = delta.clip(lower=0)
loss     = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs       = avg_gain / avg_loss
gold["rsi_14"] = 100 - (100 / (1 + rs))
 
gold["log_return"] = np.log(gold["close"] / gold["close"].shift(1))

gold["target_vol_5d"] = gold["log_return"].rolling(VOL_WINDOW).std().shift(VOL_SHIFT)
 
gold = gold.dropna()
print("Shape após feature engineering:", gold.shape)
print("NaN restantes:", gold.isna().sum().sum())


# In[167]:


# Fit GARCH no treino completo e extrair volatilidade condicional
log_ret = gold["log_return"].dropna() * 100
garch_full = arch_model(log_ret, vol="Garch", p=1, q=1)
res_full = garch_full.fit(disp="off")

gold_aligned = gold[gold["log_return"].notna()].copy()
gold_aligned["garch_vol"] = res_full.conditional_volatility.values / 100

gold = gold.merge(
    gold_aligned[["timestamp", "garch_vol"]],
    on="timestamp", how="left"
)


# # Célula 10 - Target

# In[168]:


TARGET = "target_vol_5d"
 
to_drop = [
    "currency", "unit", "timestamp", "headlines",
    "tomorrow_close", "target_return", "log_return",
    "close"
]
 
# MELHORIA 2: removidas features em nível absoluto de preço
cols_to_remove = [
    # Nível absoluto de preço -> leakage indireto via autocorrelação do close
    "sma_5",         # substituída por sma_ratio
    "sma_20",        # substituída por sma_ratio
    "bb_upper",      # substituída por bb_position
    "bb_lower",      # substituída por bb_position
    "high", "low", "open",   # em nível — usar só close como âncora temporal
 
    # Redundâncias conhecidas
    "wti_crude_usd",
    "petrodolar_wti",
    "usd_dxy_index",   # duplicata do dxy
    "max_diff",
    "silver_low",
    "platinum_volume",
    "palladium_volume",
]
 
X = gold.drop(columns=[c for c in to_drop + [TARGET] + cols_to_remove if c in gold.columns])
y = gold[TARGET]
 
n_features = X.shape[1]
print("Features restantes:", list(X.columns))
print("n_features:", n_features)


# In[169]:


split_idx = int(len(gold) * (1 - TEST_RATIO))
 
vix_train = gold["vix"].iloc[:split_idx].values
vix_test  = gold["vix"].iloc[split_idx:].values
y_train_bl = y.iloc[:split_idx].values
y_test_bl  = y.iloc[split_idx:].values
 
# Regressão linear simples VIX → target_vol
ridge_bl = Ridge(alpha=1.0)
ridge_bl.fit(vix_train.reshape(-1, 1), y_train_bl)
vix_pred_test = ridge_bl.predict(vix_test.reshape(-1, 1))
 
r2_vix  = r2_score(y_test_bl, vix_pred_test)
mae_vix = mean_absolute_error(y_test_bl, vix_pred_test)
print(f"\n── Baseline VIX -> vol ──")
print(f"  R²  (teste): {r2_vix:.4f}")
print(f"  MAE (teste): {mae_vix:.6f}")
print(f"  Correlação : {np.corrcoef(y_test_bl, vix_pred_test)[0,1]:.4f}")
print("  (O LSTM precisa superar esse baseline para justificar sua complexidade)")


# # Célula 11 - Create sequences

# In[170]:


# Célula 7 — Sequences + split
def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y.iloc[i + window_size])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X.to_numpy(), y, WINDOW_SIZE)

mask  = ~np.isnan(y_seq)
X_seq = X_seq[mask]
y_seq = y_seq[mask]

print("NaN em y_seq após mask:", np.isnan(y_seq).sum())

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_seq, y_seq, test_size=TEST_RATIO, shuffle=False
)

print("Train:", X_train_full.shape, "Test:", X_test.shape)


# # Célula 12 - Criação do modelo

# In[171]:


def build_model(learning_rate=0.0001):
    model = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate), loss="huber")
    return model


# # Célula 13 - Normalização

# In[172]:


x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(
    X_train_full.reshape(-1, n_features)
).reshape(X_train_full.shape)
X_test_scaled = x_scaler.transform(
    X_test.reshape(-1, n_features)
).reshape(X_test.shape)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train_full.reshape(-1, 1)).ravel()

print("X_train_scaled:", X_train_scaled.shape)
print("y_train_scaled mean:", y_train_scaled.mean(), "std:", y_train_scaled.std())


# # Célula 14 - Separação em treino e teste

# In[173]:


tscv = TimeSeriesSplit(n_splits=5)
folds = list(tscv.split(X_seq))   # materializa todos os folds
 
walk_r2s = []
val_X_final, val_y_final = None, None  # será o último fold — usado no treino final
 
for fold, (train_idx, val_idx) in enumerate(folds):
    X_tr = X_seq[train_idx];  X_va = X_seq[val_idx]
    y_tr = y_seq[train_idx];  y_va = y_seq[val_idx]
 
    sc_x = StandardScaler()
    X_tr_s = sc_x.fit_transform(X_tr.reshape(-1, n_features)).reshape(X_tr.shape)
    X_va_s = sc_x.transform(X_va.reshape(-1, n_features)).reshape(X_va.shape)
 
    sc_y = StandardScaler()
    y_tr_s = sc_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_va_s = sc_y.transform(y_va.reshape(-1, 1)).ravel()
 
    tf.keras.backend.clear_session()
    m = build_model()
    m.fit(
        X_tr_s, y_tr_s,
        epochs=100, batch_size=32,
        validation_data=(X_va_s, y_va_s),   # <- explícito, não validation_split
        shuffle=False, verbose=0,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True, min_delta=1e-5)]
    )
 
    pred_s = m.predict(X_va_s, verbose=0)
    pred   = sc_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    r2 = r2_score(y_va, pred)
    walk_r2s.append(r2)
    print(f"Fold {fold+1} R²: {r2:.4f}  |  pred_std: {pred.std():.6f}  |  real_std: {y_va.std():.6f}")
 
    if fold == len(folds) - 1:
        val_X_final = X_va_s.copy()
        val_y_final = y_va_s.copy()
 
print(f"\nR² médio walk-forward: {np.mean(walk_r2s):.4f}")


# # Célula 15 - Treino

# In[174]:


tf.keras.backend.clear_session()
final_model = build_model()
 
early_stop = EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True, min_delta=1e-5
)
 
history = final_model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(val_X_final, val_y_final),  
    epochs=300,
    batch_size=64,
    callbacks=[early_stop],
    shuffle=False,
    verbose=1,
)


# In[175]:


print("y_train_scaled mean:", y_train_scaled.mean())
print("y_train_scaled std: ", y_train_scaled.std())
print("y_train_scaled min: ", y_train_scaled.min())
print("y_train_scaled max: ", y_train_scaled.max())
print("NaN em y_train_scaled:", np.isnan(y_train_scaled).sum())
print("NaN em X_train_scaled:", np.isnan(X_train_scaled).sum())


# In[176]:


nan_cols = X.columns[X.isna().any()].tolist()
print("Colunas com NaN:", nan_cols)
print(X[nan_cols].isna().sum())


# # Célula 16 - Métricas

# In[177]:


y_pred_train = y_scaler.inverse_transform(
    final_model.predict(X_train_scaled)
).flatten()
y_pred_test = y_scaler.inverse_transform(
    final_model.predict(X_test_scaled)
).flatten()
 
print("Pred treino std:", y_pred_train.std())
print("Real treino std:", y_train_full.std())
print("Pred teste  std:", y_pred_test.std())
print("Real teste  std:", y_test.std())
 
for name, y_true, y_pred in [
    ("TRAIN", y_train_full, y_pred_train),
    ("TEST",  y_test,       y_pred_test),
]:
    print(f"\n{'='*60}\n{name} METRICS\n{'='*60}")
    print("MAE:  ", mean_absolute_error(y_true, y_pred))
    print("RMSE: ", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2:   ", r2_score(y_true, y_pred))
    baseline = np.full_like(y_true, fill_value=y_train_full.mean())
    print("R2 baseline (média):", r2_score(y_true, baseline))
    print("R2 baseline (VIX)  :", r2_vix, "<- barra real a superar")
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    print("Correlação pred vs real:", corr)


# In[178]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
axes[0].plot(history.history["loss"],     label="Train")
axes[0].plot(history.history["val_loss"], label="Val")
axes[0].set_yscale("log")
axes[0].set_title("Training History")
axes[0].legend()
 
axes[1].plot(y_test,                label="Real")
axes[1].plot(y_pred_test.flatten(), label="Predicted")
axes[1].set_title("Real vs Predicted (teste)")
axes[1].legend()
 
errors = y_test - y_pred_test.flatten()
axes[2].hist(errors, bins=50)
axes[2].set_title("Prediction Errors")
 
plt.tight_layout()
plt.show()
 
print("\n── Correlações com o target (novo) ──")
corr_df = pd.DataFrame(X.values, columns=X.columns)
corr_df["target"] = y.values[:len(corr_df)]
correlations = corr_df.corr()["target"].drop("target").sort_values(key=abs, ascending=False)
print(correlations.head(15))


# In[179]:


print(pd.Series(y).autocorr(1))
print(pd.Series(y).autocorr(5))
print(pd.Series(y).autocorr(20))


# In[180]:


# Célula 12 — Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['loss'],     label='Train')
axes[0].plot(history.history['val_loss'], label='Val')
axes[0].set_yscale('log')
axes[0].set_title("Training History")
axes[0].legend()

axes[1].plot(y_test,                label="Real")
axes[1].plot(y_pred_test.flatten(), label="Predicted")
axes[1].set_title("Real vs Predicted")
axes[1].legend()

errors = y_test - y_pred_test.flatten()
axes[2].hist(errors, bins=50)
axes[2].set_title("Prediction Errors")

plt.tight_layout()
plt.show()


# In[181]:


# Célula 13 — Diagnóstico de correlações
corr = pd.DataFrame(X.values, columns=X.columns)
corr["target"] = y.values[:len(corr)]
correlations = corr.corr()["target"].drop("target").sort_values(key=abs, ascending=False)
correlations


# In[182]:


print("Período completo:", gold["timestamp"].min(), "→", gold["timestamp"].max())
print("Linhas total:", len(gold))
print("\nÚltimas datas do treino (aprox):")
print("Treino vai até ~", gold["timestamp"].iloc[int(len(gold) * 0.9)])
print("Teste começa em ~", gold["timestamp"].iloc[int(len(gold) * 0.9)])


# In[183]:


y_train_full.std(), y_test.std()


# In[184]:


df = pd.DataFrame({
    "Return": y_train_full.flatten()
})
df["Return"].autocorr(lag=1)


# In[185]:


df["Return"].autocorr(lag=5)


# In[186]:


df["Return"].autocorr(lag=10)


# In[187]:


df["Return"].autocorr(lag=20)


# In[188]:


y.describe()


# In[189]:


print(y.std())


# In[190]:


print("Pred treino std:", y_pred_train.std())
print("Pred teste std:", y_pred_test.std())

print("Real treino std:", y_train_full.std())
print("Real teste std:", y_test.std())


# In[191]:


correlations


# In[192]:


for col in externos.columns:
    if col != "timestamp":
        first_valid = externos[col].first_valid_index()

        if first_valid is not None:
            print(
                col,
                externos.loc[first_valid, "timestamp"]
            )


# In[193]:


print("y_train")
print(y_tr.mean(), y_tr.std())

print("\ny_pred")
print(pred.mean(), pred.std())


# In[194]:


from arch import arch_model
log_ret = gold["log_return"].dropna() * 100  
garch = arch_model(log_ret, vol="Garch", p=1, q=1)
res = garch.fit(disp="off")
print(res.summary())


# In[ ]:


from arch import arch_model
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

log_ret = gold["log_return"].dropna() * 100

n_test = int(len(log_ret) * TEST_RATIO)
train_ret = log_ret.iloc[:-n_test]
test_ret  = log_ret.iloc[-n_test:]

preds = []
for i in range(len(test_ret)):
    window = log_ret.iloc[:len(train_ret) + i]
    m = arch_model(window, vol="Garch", p=1, q=1)
    r = m.fit(disp="off", show_warning=False)
    f = r.forecast(horizon=1)
    preds.append(np.sqrt(f.variance.values[-1, 0]) / 100)

realized = test_ret.values / 100
realized_vol = pd.Series(realized).rolling(5).std().shift(-5).dropna().values
preds_aligned = np.array(preds[:len(realized_vol)])

print("GARCH R² no teste:", r2_score(realized_vol, preds_aligned))

