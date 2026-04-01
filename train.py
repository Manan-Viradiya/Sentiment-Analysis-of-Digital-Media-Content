"""
train.py
--------
Fine-tunes DistilBERT on the YouTube Comments sentiment dataset and saves
the model + tokenizer to ./models/sentiment_model/ so that sentiment_engine.py
can load them at inference time.

Usage
-----
    # Default — expects CSV at ./data/YoutubeCommentsDataSet.csv
    python train.py

    # Custom CSV path
    python train.py --data ./data/my_comments.csv

    # Skip training, just verify the saved model with a quick inference test
    python train.py --verify-only

Label encoding (matches sentiment_engine.py LABEL_MAP)
-------------------------------------------------------
    0  →  Negative
    1  →  Neutral
    2  →  Positive
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH = "./data/YoutubeCommentsDataSet.csv"
MODEL_SAVE_DIR    = "./models/sentiment_model/"  # single dir for model + tokenizer
CHECKPOINT_DIR    = "./models/checkpoints/"
BASE_MODEL        = "distilbert-base-uncased"

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}   # text → int
ID2LABEL  = {0: "Negative", 1: "Neutral", 2: "Positive"}   # int → display label
LABEL2ID  = {"Negative": 0, "Neutral": 1, "Positive": 2}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Data loading & preprocessing ──────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV, drop nulls , encode labels,
    and rename columns to the 'text' / 'label' convention HuggingFace expects.
    """
    log.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("Raw shape: %s", df.shape)

    df = df.dropna(subset=["Comment", "Sentiment"]).reset_index(drop=True)
    log.info("After dropna: %s rows", len(df))

    # Normalise casing before mapping (handles 'Positive', 'POSITIVE', etc.)
    df["Sentiment"] = df["Sentiment"].str.strip().str.lower()

    unmapped = set(df["Sentiment"].unique()) - set(LABEL_MAP.keys())
    if unmapped:
        log.warning("Unknown sentiment values (will be dropped): %s", unmapped)
        df = df[df["Sentiment"].isin(LABEL_MAP.keys())].reset_index(drop=True)

    df["label"] = df["Sentiment"].map(LABEL_MAP)
    df = df.rename(columns={"Comment": "text"})[["text", "label"]]

    log.info("Label distribution:\n%s", df["label"].value_counts().to_string())
    return df


def split_and_save(df: pd.DataFrame):
    """Stratified 70 / 30 train-test split; saves CSVs for reference."""
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["label"],
    )
    os.makedirs("./data/splits", exist_ok=True)
    train_df.to_csv("./data/splits/train.csv", index=False)
    test_df.to_csv("./data/splits/test.csv",  index=False)
    log.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ── Tokenisation ──────────────────────────────────────────────────────────────

def build_tokenized_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
) -> DatasetDict:
    """Convert DataFrames → HuggingFace DatasetDict and tokenise."""
    datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test":  Dataset.from_pandas(test_df),
    })

    def tokenize(batch):
        texts = [str(t) for t in batch["text"]]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

    tokenized = datasets.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["text"])

    tokenized.set_format("torch")

    return tokenized


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="weighted")
    return {"accuracy_score": acc, "f1_score": f1}


# ── Training ──────────────────────────────────────────────────────────────────

def train(csv_path: str):
    # 1. Data
    df = load_and_clean(csv_path)
    train_df, test_df = split_and_save(df)

    # 2. Tokenizer
    log.info("Loading tokenizer: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 3. Tokenise
    tokenized = build_tokenized_datasets(train_df, test_df, tokenizer)

    # 4. Model — pass id2label/label2id so the pipeline can decode labels
    log.info("Loading base model: %s", BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("Training on: %s", device)

    # 5. Training arguments
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        greater_is_better=True,
        report_to="none",
        fp16=use_fp16,
        logging_steps=50,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    log.info("Starting training …")
    trainer.train()

    # 8. Evaluate
    metrics = trainer.evaluate()
    log.info("Final evaluation: %s", metrics)

    # 9. Save
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    log.info("Model + tokenizer saved to %s", MODEL_SAVE_DIR)

    return metrics


# ── Quick inference smoke-test ────────────────────────────────────────────────

def verify():
    """
    Load the saved model and run three test comments through it.
    """
    log.info("Running inference smoke-test from %s …", MODEL_SAVE_DIR)

    if not os.path.isdir(MODEL_SAVE_DIR):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_SAVE_DIR}'. Run training first."
        )

    device = 0 if torch.cuda.is_available() else -1

    clf = pipeline(
        "text-classification",
        model=MODEL_SAVE_DIR,
        tokenizer=MODEL_SAVE_DIR,
        device=device,
        truncation=True,
        max_length=128,
    )

    test_comments = [
        "This is an amazing tutorial, very helpful!",
        "I am quite confused by this explanation.",
        "It's okay, neither good nor bad.",
    ]

    print("\n" + "=" * 60)
    print("  INFERENCE SMOKE-TEST")
    print("=" * 60)
    for comment, pred in zip(test_comments, clf(test_comments)):
        raw_label = pred["label"]       # e.g. "Negative" — set by id2label
        score     = pred["score"]
        print(f"\n  Comment   : {comment}")
        print(f"  Sentiment : {raw_label}  (confidence: {score:.4f})")
    print("=" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DistilBERT sentiment model on YouTube comments."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the CSV file (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip training; just run the inference smoke-test on the saved model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.verify_only:
        verify()
    else:
        metrics = train(args.data)
        verify()
        print("\nTraining complete. Summary:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
