"""Text normalisation and train/validation/test preparation.

Two text views are produced for every article:
  * ``text``: HTML residue and news-agency tags removed, casing and punctuation kept.
    The BERT tokenizer consumes this view.
  * ``clean_text``: additionally lower-cased and reduced to alphabetic tokens.
    The TF-IDF model consumes this view.
"""
import html
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

# AG News descriptions carry escaped entities such as "#39;" or "quot;" and
# agency bylines such as "(Reuters)" or "AFP -". These leak the source rather
# than the topic, so we strip them before modelling.
_ENTITY = re.compile(r"&?#\d+;|&?(quot|amp|lt|gt|nbsp);", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>|&lt;.*?&gt;", re.IGNORECASE)
_SOURCES = (
    r"AP|AFP|Reuters|Canadian Press|CP|SPACE\.com|Forbes\.com|NewsFactor|"
    r"Ziff Davis|TechWeb|PC World|washingtonpost\.com|USATODAY\.com"
)
_AGENCY = re.compile(rf"\((?:{_SOURCES})\)|\b(?:{_SOURCES})\s+-\s+", re.IGNORECASE)


def normalize_text(text):
    text = html.unescape(str(text))
    text = _TAG.sub(" ", text)
    text = _ENTITY.sub(" ", text)
    text = _AGENCY.sub(" ", text)
    text = text.replace("\\", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text):
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare(df):
    out = pd.DataFrame({"label": df["label"].astype(int)})
    out["text"] = df["text"].map(normalize_text)
    out["clean_text"] = df["text"].map(clean_text)
    out["label_name"] = out["label"].map(config.ID2LABEL)
    return out


def preprocess_and_save(raw_dir=config.RAW_DIR, out_dir=config.PROCESSED_DIR):
    train_raw = pd.read_csv(raw_dir / "ag_news_train.csv")
    test_raw = pd.read_csv(raw_dir / "ag_news_test.csv")

    train_full = prepare(train_raw)
    test_df = prepare(test_raw)

    # The official test split is kept untouched; validation data is drawn
    # from the training split with stratification on the label.
    train_df, val_df = train_test_split(
        train_full,
        test_size=config.VAL_FRACTION,
        stratify=train_full["label"],
        random_state=config.SEED,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        path = out_dir / f"ag_news_clean_{name}.csv"
        frame.to_csv(path, index=False)
        print(f"{name}: {len(frame):,} rows -> {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    preprocess_and_save()
