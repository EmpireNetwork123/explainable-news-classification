"""Shared paths, label definitions and hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

# AG News label ids as distributed on the Hugging Face Hub.
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
ID2LABEL = dict(enumerate(LABEL_NAMES))
LABEL2ID = {name: i for i, name in ID2LABEL.items()}

HF_DATASET = "fancyzhx/ag_news"
SEED = 42
VAL_FRACTION = 0.1

# TF-IDF + Multinomial Naive Bayes
NB_ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0]

# BERT fine-tuning
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 128
BERT_EPOCHS = 3
BERT_BATCH_SIZE = 32
BERT_LEARNING_RATE = 2e-5
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.06


def ensure_dirs():
    for d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
