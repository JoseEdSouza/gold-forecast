""" "
gold_embeddings.py
==================
Generates embeddings from news headlines for gold price forecasting.

Strategy:
  - For each day t, concatenate headlines from the 1d, 7d, and 30d windows
  - Generate embeddings with nomic-embed-text-v1 (local, GPU) and/or OpenAI (async)
  - Save the result as CSV with columns emb_*_dim0 ... emb_*_dimN

Expected input CSV structure:
  - Date column      : "Date" (YYYY-MM-DD format or similar)
  - Price column     : "Price" (or Open, Close, etc.)
  - Headlines column : "News" (daily headlines already concatenated)

Usage:
  python gold_embeddings.py --input gold.csv --backend nomic
  python gold_embeddings.py --input gold.csv --backend openai --openai-key sk-...
  python gold_embeddings.py --input gold.csv --backend both   --openai-key sk-...

Optional flags:
  --batch-size 256        nomic batch size (default: 256 on GPU, 64 on CPU)
  --openai-concurrency 5  concurrent OpenAI requests (default: 5)
  --device cuda           force GPU for nomic; "cpu" forces CPU
"""

import argparse
import asyncio
import logging
import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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

    logger.info("Loading nomic-embed-text-v1 on device: %s", device)
    logger.info("(first run downloads ~550MB)")

    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1",
        trust_remote_code=True,
        device=device,
        token=hf_token,
    )
    logger.info("Model loaded. Active device: %s", model.device)
    return model


def embed_nomic(texts: list[str], model, batch_size: int = 256) -> np.ndarray:
    """
    Generates embeddings with the local nomic model.

    Nomic requires the "search_document: " prefix on each text — without it,
    embedding quality drops significantly.

    batch_size:
      - 6GB GPU: 256–512 works well for short headlines
      - CPU:     64 is safer
    """
    prefixed = [f"search_document: {t}" for t in texts]

    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-norm → cosine similarity = dot product
        convert_to_numpy=True,
    )
    return embeddings  # shape: (N, 768)


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
        logger.info("  Window %dd: %d texts built.", W, len(windowed))

    return result


# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
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
) -> pd.DataFrame:
    """
    Runs the full pipeline and returns the DataFrame with embedding columns.
    """
    logger.info("=== Building temporal windows ===")
    windowed_texts = build_windowed_texts(df, news_col, windows)

    result_df = df.copy()

    # ── Nomic (local, GPU) ─────────────────────────────────────
    if backend in ("nomic", "both"):
        nomic_model = load_nomic_model(device=device, hf_token=hf_token)

        # Adjust batch_size automatically when running on CPU
        effective_batch = nomic_batch_size
        if str(nomic_model.device) == "cpu" and nomic_batch_size > 64:
            effective_batch = 64
            logger.info("  CPU detected: batch_size adjusted to %d.", effective_batch)

        logger.info("=== Generating embeddings with nomic-embed-text-v1 (batch=%d) ===", effective_batch)

        for W in windows:
            logger.info("  Window %dd...", W)
            embs = embed_nomic(
                windowed_texts[W], nomic_model, batch_size=effective_batch
            )

            emb_df = pd.DataFrame(
                embs,
                columns=[f"emb_nomic_{W}d_dim{i}" for i in range(embs.shape[1])],
                index=result_df.index,
            )
            result_df = pd.concat([result_df, emb_df], axis=1)
            logger.info("  %d dims added — window %dd.", embs.shape[1], W)

    # ── OpenAI (API, async) ────────────────────────────────────
    if backend in ("openai", "both"):
        if not openai_key:
            raise ValueError("Provide --openai-key to use the OpenAI backend.")

        logger.info("=== Generating embeddings with OpenAI %s ===", openai_model)
        logger.info("    batch_size=%d, concurrency=%d", openai_batch_size, openai_concurrency)

        model_tag = openai_model.replace("text-embedding-", "").replace("-", "_")

        for W in windows:
            logger.info("  Window %dd...", W)
            embs = embed_openai(
                windowed_texts[W],
                api_key=openai_key,
                model=openai_model,
                batch_size=openai_batch_size,
                max_concurrent=openai_concurrency,
            )

            emb_df = pd.DataFrame(
                embs,
                columns=[f"emb_{model_tag}_{W}d_dim{i}" for i in range(embs.shape[1])],
                index=result_df.index,
            )
            result_df = pd.concat([result_df, emb_df], axis=1)
            logger.info("  %d dims added — window %dd.", embs.shape[1], W)

    return result_df


# ─────────────────────────────────────────────
# 5. EXTRA FEATURE: 1d vs 7d DEVIATION
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
        cols_1d = [c for c in df.columns if c.startswith(f"emb_{tag}_1d_dim")]
        cols_7d = [c for c in df.columns if c.startswith(f"emb_{tag}_7d_dim")]

        if not cols_1d or not cols_7d:
            logger.warning("  Deviation %s: 1d or 7d windows not found, skipping.", tag)
            continue

        deviation = df[cols_1d].values - df[cols_7d].values
        n_dims = deviation.shape[1]

        dev_df = pd.DataFrame(
            deviation,
            columns=[f"emb_{tag}_dev_1d_7d_dim{i}" for i in range(n_dims)],
            index=df.index,
        )
        df = pd.concat([df, dev_df], axis=1)
        logger.info("  1d-7d deviation added for %s (%d dims).", tag, n_dims)

    return df


# ─────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from news headlines for gold price forecasting."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--date-col", default="Date")
    parser.add_argument("--news-col", default="News")
    parser.add_argument(
        "--backend", default="nomic", choices=["nomic", "openai", "both"]
    )
    parser.add_argument(
        "--device", default="auto", help="Device for nomic: 'auto', 'cuda', 'cpu'"
    )
    parser.add_argument(
        "--batch-size",
        default=256,
        type=int,
        help="Nomic batch size (default 256; auto-reduced on CPU)",
    )
    parser.add_argument("--openai-key", default=None)
    parser.add_argument(
        "--openai-model",
        default="text-embedding-3-small",
        choices=["text-embedding-3-small", "text-embedding-3-large"],
    )
    parser.add_argument(
        "--openai-concurrency",
        default=5,
        type=int,
        help="Concurrent requests to OpenAI (default: 5)",
    )
    parser.add_argument(
        "--windows",
        default="1,7,30",
        help="Comma-separated window sizes in days (default: 1,7,30)",
    )
    args = parser.parse_args()

    # ── Load env vars (.env takes precedence over system env) ──
    load_dotenv()
    openai_key = args.openai_key or os.getenv("OPENAI_KEY")
    hf_token = os.getenv("HF_TOKEN")

    # ── Load and sort chronologically ──────────────────────────
    logger.info("Loading %s...", args.input)
    df = pd.read_csv(args.input, parse_dates=[args.date_col])
    df = df.sort_values(args.date_col).reset_index(drop=True)
    logger.info(
        "  %d rows | %s → %s",
        len(df),
        df[args.date_col].min().date(),
        df[args.date_col].max().date(),
    )

    windows = [int(w) for w in args.windows.split(",")]
    logger.info("  Windows: %s", windows)

    # ── Embedding pipeline ─────────────────────────────────────
    result_df = run_pipeline(
        df=df,
        news_col=args.news_col,
        windows=windows,
        backend=args.backend,
        device=args.device,
        nomic_batch_size=args.batch_size,
        hf_token=hf_token,
        openai_key=openai_key,
        openai_model=args.openai_model,
        openai_concurrency=args.openai_concurrency,
    )

    # ── 1d vs 7d deviation (if both windows were generated) ────
    if 1 in windows and 7 in windows:
        logger.info("=== Computing 1d vs 7d deviation ===")
        result_df = add_deviation_features(result_df, args.backend, args.openai_model)

    # ── Save ───────────────────────────────────────────────────
    output_path = args.output or args.input.replace(".csv", "_embeddings.csv")
    result_df.to_csv(output_path, index=False)

    n_emb_cols = len([c for c in result_df.columns if c.startswith("emb_")])
    logger.info("Saved to: %s", output_path)
    logger.info("  Final shape: %s (%d embedding columns)", result_df.shape, n_emb_cols)


if __name__ == "__main__":
    main()
