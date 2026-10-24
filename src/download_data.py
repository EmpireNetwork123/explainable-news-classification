"""Download the AG News corpus and store the official splits as CSV."""
from datasets import load_dataset

from src import config


def download():
    config.ensure_dirs()
    dataset = load_dataset(config.HF_DATASET)
    for split in ("train", "test"):
        out = config.RAW_DIR / f"ag_news_{split}.csv"
        dataset[split].to_pandas().to_csv(out, index=False)
        print(f"{split}: {len(dataset[split]):,} rows -> {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    download()
