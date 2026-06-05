"""
gold_embeddings.py
==================
Generates embeddings from news headlines for gold price forecasting.

Strategy:
  - For each day t, concatenate headlines from the 1d, 7d, and 30d windows
  - Generate embeddings with nomic-embed-text-v1 (local, GPU) and/or OpenAI (async)
  - Save the result as Parquet (zstd) with one list[float] column per (backend, window)

Expected input CSV structure:
  - Date column      : "Date" (YYYY-MM-DD format or similar)
  - Price column     : "Price" (or Open, Close, etc.)
  - Headlines column : "News" (daily headlines already concatenated)

Usage:
  Edit the Config block at the bottom of this file, then run:
    python gold_embeddings.py
"""

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. NOMIC — local, GPU-aware
# ─────────────────────────────────────────────


def load_nomic_model(device: str = "auto", hf_token: str | None = None):
    """
    Loads nomic-embed-text-v1 via sentence-transformers.

    device: "auto" detects GPU automatically, "cuda" forces GPU, "cpu" forces CPU.
    """
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("─── nomic-embed-text-v1 ────────────────────────────")
    logger.info("    device : %s", device)
    logger.info("    note   : first run downloads ~550 MB")

    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1",
        trust_remote_code=True,
        device=device,
        token=hf_token,
    )
    logger.info("    ✓ loaded on %s", model.device)
    return model


def embed_nomic(texts: list[str], model, batch_size: int = 256) -> np.ndarray:
    """
    Generates embeddings with the local nomic model.

    Nomic requires the "search_document: " prefix on each text — without it,
    embedding quality drops significantly.

    On CUDA OOM the batch_size is halved automatically until encode succeeds.
    """
    prefixed = [f"search_document: {t}" for t in texts]

    while batch_size >= 1:
        try:
            return model.encode(
                prefixed,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,  # L2-norm → cosine similarity = dot product
                convert_to_numpy=True,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size //= 2
            logger.warning("    CUDA OOM — retrying with batch_size=%d", batch_size)

    raise RuntimeError("embed_nomic: OOM even with batch_size=1")


# ─────────────────────────────────────────────
# 2. OPENAI — async with concurrency control
# ─────────────────────────────────────────────


async def _embed_batch_async(
    client: AsyncOpenAI, texts: list[str], model: str, semaphore: asyncio.Semaphore
) -> list[list[float]]:
    """
    Makes a single asynchronous request to the OpenAI API.
    The semaphore limits how many requests run at the same time,
    preventing rate limit issues.
    """
    async with semaphore:
        response = await client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]


async def embed_openai_async(
    texts: list[str],
    api_key: str,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_concurrent: int = 5,
) -> np.ndarray:
    """
    Generates embeddings through the OpenAI API asynchronously.

    max_concurrent controls how many simultaneous requests are made.
    With 5 parallel requests of 100 texts each, you process
    500 texts per "round" without hitting the rate limit.

    For 6k texts (3 windows × 6k rows = 18k total calls,
    split into batches of 100 = 180 requests), async reduces
    runtime from ~3 min (sequential) to ~40s.
    """
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Split into batches
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    # Launch all coroutines with concurrency control
    tasks = [_embed_batch_async(client, batch, model, semaphore) for batch in batches]
    results = await atqdm.gather(*tasks, desc=f"OpenAI {model} (async)")

    # Flatten: list of lists -> array (N, dims)
    flat = [emb for batch_result in results for emb in batch_result]
    return np.array(flat)


def embed_openai(
    texts: list[str],
    api_key: str,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    max_concurrent: int = 5,
) -> np.ndarray:
    """
    Synchronous wrapper for embed_openai_async.
    Handles the event loop correctly regardless of the environment
    (script, Jupyter, etc.).
    """
    try:
        # Inside an already running event loop (for example, Jupyter)
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                embed_openai_async(texts, api_key, model, batch_size, max_concurrent),
            )
            return future.result()
    except RuntimeError:
        # No event loop running (regular script)
        return asyncio.run(
            embed_openai_async(texts, api_key, model, batch_size, max_concurrent)
        )


# ─────────────────────────────────────────────
# 3. TEMPORAL WINDOW CONSTRUCTION
# ─────────────────────────────────────────────


def build_windowed_texts(
    df: pd.DataFrame, news_col: str, windows: list[int]
) -> dict[int, list[str]]:
    """
    For each window W and each day t, concatenates headlines from the W
    previous days (including day t), separated by " | ".

    Example for W=3, day t=5:
      "day3_headlines | day4_headlines | day5_headlines"

    For the first days in the dataset (t < W), it uses all available days
    instead of dropping the row.

    Returns: {1: [...], 7: [...], 30: [...]}
    """
    news = df[news_col].fillna("").tolist()
    result = {}

    for W in windows:
        windowed = []
        for t in range(len(news)):
            start = max(0, t - W + 1)
            window_text = " | ".join(news[start : t + 1])
            windowed.append(window_text)
        result[W] = windowed
        logger.info("    window %2dd → %d texts", W, len(windowed))

    return result


# ─────────────────────────────────────────────
# 4. EMBEDDING CACHE
# ─────────────────────────────────────────────


def _cache_key(texts: list[str], tag: str, W: int) -> str:
    """SHA-1 over the windowed texts + model tag + window size."""
    h = hashlib.sha1()
    h.update(tag.encode())
    h.update(str(W).encode())
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()[:16]


def _cache_path(cache_dir: str, tag: str, W: int, key: str) -> Path:
    return Path(cache_dir) / f"{tag}_{W}d_{key}.parquet"


def _load_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        logger.info("    ✓ cache hit  → %s", path.name)
        return pd.read_parquet(path)
    return None


def _save_cache(emb_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(path, index=True, compression="zstd")
    logger.info("    cached      → %s", path.name)


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────


def run_pipeline(
    df: pd.DataFrame,
    news_col: str,
    windows: list[int],
    backend: str,
    device: str = "auto",
    nomic_batch_size: int = 256,
    hf_token: str | None = None,
    openai_key: str = None,
    openai_model: str = "text-embedding-3-small",
    openai_batch_size: int = 100,
    openai_concurrency: int = 5,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """
    Runs the full pipeline and returns the DataFrame with embedding columns.
    """
    logger.info("─── temporal windows ───────────────────────────────")
    windowed_texts = build_windowed_texts(df, news_col, windows)

    result_df = df.copy()

    # ── Nomic (local, GPU) ─────────────────────────────────────
    if backend in ("nomic", "both"):
        logger.info("")
        nomic_model = load_nomic_model(device=device, hf_token=hf_token)

        # Adjust batch_size automatically when running on CPU
        effective_batch = nomic_batch_size
        if str(nomic_model.device) == "cpu" and nomic_batch_size > 64:
            effective_batch = 64
            logger.info("    CPU detected — batch_size reduced to %d", effective_batch)

        logger.info("")
        logger.info("─── nomic embeddings ───────────────────────────────")
        logger.info("    batch : %d", effective_batch)

        for W in windows:
            logger.info("    window %2dd ...", W)
            cached = None
            if cache_dir:
                key = _cache_key(windowed_texts[W], "nomic", W)
                path = _cache_path(cache_dir, "nomic", W, key)
                cached = _load_cache(path)

            if cached is not None:
                emb_df = cached
            else:
                embs = embed_nomic(
                    windowed_texts[W], nomic_model, batch_size=effective_batch
                )
                emb_df = pd.DataFrame(
                    {f"emb_nomic_{W}d": list(embs.tolist())},
                    index=result_df.index,
                )
                if cache_dir:
                    _save_cache(emb_df, path)

            result_df = pd.concat([result_df, emb_df], axis=1)
            logger.info(
                "    ✓ %dd  →  %d dims", W, len(emb_df[f"emb_nomic_{W}d"].iloc[0])
            )

    # ── OpenAI (API, async) ────────────────────────────────────
    if backend in ("openai", "both"):
        if not openai_key:
            raise ValueError("Provide --openai-key to use the OpenAI backend.")

        logger.info("")
        logger.info("─── openai embeddings ──────────────────────────────")
        logger.info("    model       : %s", openai_model)
        logger.info("    batch       : %d", openai_batch_size)
        logger.info("    concurrency : %d", openai_concurrency)

        model_tag = openai_model.replace("text-embedding-", "").replace("-", "_")

        for W in windows:
            logger.info("    window %2dd ...", W)
            cached = None
            if cache_dir:
                key = _cache_key(windowed_texts[W], model_tag, W)
                path = _cache_path(cache_dir, model_tag, W, key)
                cached = _load_cache(path)

            if cached is not None:
                emb_df = cached
            else:
                embs = embed_openai(
                    windowed_texts[W],
                    api_key=openai_key,
                    model=openai_model,
                    batch_size=openai_batch_size,
                    max_concurrent=openai_concurrency,
                )
                emb_df = pd.DataFrame(
                    {f"emb_{model_tag}_{W}d": embs.tolist()},
                    index=result_df.index,
                )
                if cache_dir:
                    _save_cache(emb_df, path)

            result_df = pd.concat([result_df, emb_df], axis=1)
            logger.info(
                "    ✓ %dd  →  %d dims", W, len(emb_df[f"emb_{model_tag}_{W}d"].iloc[0])
            )

    return result_df


# ─────────────────────────────────────────────
# 6. EXTRA FEATURE: 1d vs 7d DEVIATION
# ─────────────────────────────────────────────


def add_deviation_features(
    df: pd.DataFrame,
    backend: str,
    openai_model: str = "text-embedding-3-small",
) -> pd.DataFrame:
    """
    Computes emb_1d - emb_7d per dimension for each backend.

    This deviation vector captures "narrative shift": when the day's discourse
    diverges significantly from the week's average, it may signal
    an event relevant to the gold price.
    """
    tags = []
    if backend in ("nomic", "both"):
        tags.append("nomic")
    if backend in ("openai", "both"):
        tags.append(openai_model.replace("text-embedding-", "").replace("-", "_"))

    for tag in tags:
        col_1d = f"emb_{tag}_1d"
        col_7d = f"emb_{tag}_7d"

        if col_1d not in df.columns or col_7d not in df.columns:
            logger.warning("    [%s] 1d or 7d column not found, skipping", tag)
            continue

        arr_1d = np.stack(df[col_1d].tolist())
        arr_7d = np.stack(df[col_7d].tolist())
        deviation = (arr_1d - arr_7d).tolist()

        df[f"emb_{tag}_dev_1d_7d"] = deviation
        logger.info("    ✓ [%s]  1d–7d deviation  →  %d dims", tag, len(deviation[0]))

    return df


# ─────────────────────────────────────────────
# 6. CONFIG + ENTRY POINT
# ─────────────────────────────────────────────


@dataclass
class Config:
    data_path: str = "gold.csv"
    output_dir: str = "./out"  # output directory; filename derived from data_path
    separator: str = ","  # for CSV loading (not needed for Parquet)
    date_col: str = "Date"
    news_col: str = "News"
    backend: Literal["nomic", "openai", "both"] = "nomic"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    nomic_batch_size: int = 32  # halved automatically on CUDA OOM; 64 on CPU
    openai_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = (
        "text-embedding-3-small"
    )
    openai_batch_size: int = 100
    openai_concurrency: int = 5
    windows: list[int] = field(default_factory=lambda: [1, 7, 30])
    cache_dir: str | None = ".emb_cache"  # None disables caching


def main(cfg: Config) -> None:
    # ── Load env vars ──────────────────────────────────────────
    load_dotenv()
    openai_key = os.getenv("OPENAI_KEY")
    hf_token = os.getenv("HF_TOKEN")

    # ── Load and sort chronologically ──────────────────────────
    logger.info("─── input ──────────────────────────────────────────")
    logger.info("    file    : %s", cfg.data_path)
    df = pd.read_csv(cfg.data_path, parse_dates=[cfg.date_col], sep=cfg.separator)
    df = df.sort_values(cfg.date_col).reset_index(drop=True)
    logger.info(
        "    rows    : %d  (%s → %s)",
        len(df),
        df[cfg.date_col].min().date(),
        df[cfg.date_col].max().date(),
    )
    logger.info("    windows : %s", cfg.windows)
    logger.info("    cache   : %s", cfg.cache_dir or "disabled")
    logger.info("")

    # ── Embedding pipeline ─────────────────────────────────────
    result_df = run_pipeline(
        df=df,
        news_col=cfg.news_col,
        windows=cfg.windows,
        backend=cfg.backend,
        device=cfg.device,
        nomic_batch_size=cfg.nomic_batch_size,
        hf_token=hf_token,
        openai_key=openai_key,
        openai_model=cfg.openai_model,
        openai_concurrency=cfg.openai_concurrency,
        cache_dir=cfg.cache_dir,
    )

    # ── 1d vs 7d deviation (if both windows were generated) ────
    if 1 in cfg.windows and 7 in cfg.windows:
        logger.info("")
        logger.info("─── 1d vs 7d deviation ─────────────────────────────")
        result_df = add_deviation_features(result_df, cfg.backend, cfg.openai_model)

    # ── Save ───────────────────────────────────────────────────
    stem = Path(cfg.data_path).stem + "_embeddings"
    base = Path(cfg.output_dir) / f"{stem}.parquet"
    base.parent.mkdir(parents=True, exist_ok=True)
    output_path = base
    if output_path.exists():
        n = 1
        while (candidate := base.with_stem(f"{base.stem}_{n}")).exists():
            n += 1
        output_path = candidate

    result_df.to_parquet(output_path, index=False, compression="zstd")

    emb_cols = [c for c in result_df.columns if c.startswith("emb_")]
    n_emb_cols = len(emb_cols)
    logger.info("")
    logger.info("─── done ───────────────────────────────────────────")
    logger.info("    path    : %s", output_path)
    logger.info("    shape   : %s", result_df.shape)
    logger.info("    emb cols: %d", n_emb_cols)


# ─────────────────────────────────────────────
# CONFIGURE AND RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config(
        data_path="./data/final_gold_data.csv",
        date_col="timestamp",
        news_col="headlines",
        separator=";",
        backend="nomic",
        nomic_batch_size=128,
        windows=[1, 7],
    )
    main(cfg)
