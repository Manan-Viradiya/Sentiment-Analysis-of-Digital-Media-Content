"""
sentiment_engine.py
-------------------
Handles all DistilBERT model loading and sentiment inference logic.
Loads the fine-tuned model from ./models/sentiment_model/ and
returns Positive / Negative / Neutral labels for a list of input texts.

Label mapping (matches train.py id2label config)
-------------------------------------------------
    LABEL_0  →  Negative
    LABEL_1  →  Neutral
    LABEL_2  →  Positive

    If the model was saved with id2label (as train.py does), HuggingFace will
    output "Negative" / "Neutral" / "Positive" directly — no mapping needed.
    The LABEL_MAP below handles both cases gracefully.
"""

import logging
import os
from typing import Dict, List

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH    = "./models/sentiment_model/"
BATCH_SIZE    = 32
MAX_TOKEN_LEN = 128   # matches training truncation length

# Handles both outcomes:
#   a) model saved WITH id2label → outputs "Negative"/"Neutral"/"Positive" directly
#   b) model saved WITHOUT id2label → outputs "LABEL_0"/"LABEL_1"/"LABEL_2"
LABEL_MAP: Dict[str, str] = {
    # Direct human-readable (case a — produced by train.py)
    "Negative": "Negative",
    "Neutral":  "Neutral",
    "Positive": "Positive",
    # Raw indices (case b — fallback)
    "LABEL_0":  "Negative",
    "LABEL_1":  "Neutral",
    "LABEL_2":  "Positive",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Singleton cache — model is loaded only once per session ───────────────────

_pipeline_cache = None


def _load_pipeline():
    """Load (or return cached) HuggingFace pipeline."""
    global _pipeline_cache

    if _pipeline_cache is not None:
        return _pipeline_cache

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model directory not found: '{MODEL_PATH}'.\n"
            "Run  python train.py  first to train and save the model."
        )

    logger.info("Loading tokenizer from %s ...", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    logger.info("Loading model from %s ...", MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    device = 0 if torch.cuda.is_available() else -1
    logger.info("Inference device: %s", "GPU (cuda:0)" if device == 0 else "CPU")

    _pipeline_cache = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=MAX_TOKEN_LEN,
        batch_size=BATCH_SIZE,
    )

    logger.info("Sentiment pipeline ready.")
    return _pipeline_cache


def _normalise_label(raw: str) -> str:
    """Map any raw model label to a clean display label."""
    return (
        LABEL_MAP.get(raw)
        or LABEL_MAP.get(raw.upper())
        or raw.capitalize()
    )


# ── Public API ────────────────────────────────────────────────────────────────

def predict_sentiment(text_list: List[str]) -> List[str]:
    """
    Predict sentiment for a list of raw comment strings.

    Parameters
    ----------
    text_list : list[str]
        Raw comment / post texts.

    Returns
    -------
    list[str]
        One of "Positive", "Negative", or "Neutral" per input string.

    Raises
    ------
    FileNotFoundError
        If ./models/sentiment_model/ does not exist.
    RuntimeError
        If inference fails.
    """
    if not text_list:
        return []

    nlp = _load_pipeline()
    cleaned = [t.strip() if isinstance(t, str) and t.strip() else "[empty]"
               for t in text_list]

    try:
        raw_results = nlp(cleaned)
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc

    return [_normalise_label(r["label"]) for r in raw_results]


def predict_sentiment_with_scores(text_list: List[str]) -> List[Dict]:
    """
    Same as predict_sentiment but also returns the confidence score.

    Returns
    -------
    list[dict]
        [{"label": "Positive", "score": 0.97}, ...]
    """
    if not text_list:
        return []

    nlp = _load_pipeline()
    cleaned = [t.strip() if isinstance(t, str) and t.strip() else "[empty]"
               for t in text_list]

    try:
        raw_results = nlp(cleaned)
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc

    return [
        {
            "label": _normalise_label(r["label"]),
            "score": round(r["score"], 4),
        }
        for r in raw_results
    ]
