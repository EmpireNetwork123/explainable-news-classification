"""Fine-tune BERT (bert-base-uncased) on AG News.

The checkpoint with the highest validation accuracy is restored at the end of
training and evaluated once on the official test split.

Example:
    python -m src.train_bert
    python -m src.train_bert --max-train-samples 2000 --epochs 1   # quick CPU check
"""
import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src import config
from src.utils import compute_metrics, plot_confusion_matrix, plot_training_curve, save_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default=config.BERT_MODEL_NAME,
                   help="Hub id or local directory of a pre-trained BERT checkpoint.")
    p.add_argument("--epochs", type=float, default=config.BERT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BERT_BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=config.BERT_LEARNING_RATE)
    p.add_argument("--max-length", type=int, default=config.BERT_MAX_LENGTH)
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Subsample the training split (for smoke tests).")
    p.add_argument("--max-eval-samples", type=int, default=None,
                   help="Subsample the validation and test splits (for smoke tests).")
    p.add_argument("--output-dir", default=str(config.MODELS_DIR / "bert"))
    return p.parse_args()


def load_split(name, limit=None):
    df = pd.read_csv(config.PROCESSED_DIR / f"ag_news_clean_{name}.csv")
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=config.SEED)
    return df[["text", "label"]].reset_index(drop=True)


def tokenize_split(df, tokenizer, max_length):
    ds = Dataset.from_pandas(df)
    return ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def accuracy_fn(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}


def main():
    args = parse_args()
    config.ensure_dirs()
    set_seed(config.SEED)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(config.LABEL_NAMES),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    train_ds = tokenize_split(load_split("train", args.max_train_samples), tokenizer, args.max_length)
    val_ds = tokenize_split(load_split("val", args.max_eval_samples), tokenizer, args.max_length)
    test_df = load_split("test", args.max_eval_samples)
    test_ds = tokenize_split(test_df, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=config.BERT_WEIGHT_DECAY,
        warmup_steps=config.BERT_WARMUP_RATIO,  # a float in [0, 1) is read as a fraction of total steps
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        seed=config.SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=accuracy_fn,
    )
    trainer.train()

    # Reload the best checkpoint through from_pretrained. The in-place restore
    # performed by load_best_model_at_end skips LayerNorm parameters that are
    # serialised under their legacy gamma/beta names.
    best = trainer.state.best_model_checkpoint
    if best:
        print(f"Best checkpoint: {best}")
        trainer.model = trainer.model_wrapped = AutoModelForSequenceClassification.from_pretrained(best).to(
            trainer.args.device
        )

    final_dir = config.MODELS_DIR / "bert_final"
    trainer.save_model(final_dir)

    prediction = trainer.predict(test_ds)
    y_pred = prediction.predictions.argmax(axis=-1)
    metrics = compute_metrics(test_df["label"].to_numpy(), y_pred)
    metrics.update({
        "model": f"{args.model_name} (fine-tuned)",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "train_samples": len(train_ds),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    })

    suffix = "" if args.max_train_samples is None else "_subset"
    save_json(metrics, config.METRICS_DIR / f"bert_metrics{suffix}.json")
    save_json(trainer.state.log_history, config.METRICS_DIR / f"bert_log_history{suffix}.json")
    plot_training_curve(trainer.state.log_history, config.FIGURES_DIR / f"bert_training_curve{suffix}.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"], "Confusion Matrix: Fine-tuned BERT",
        config.FIGURES_DIR / f"bert_confusion_matrix{suffix}.png",
    )
    print(f"Test accuracy = {metrics['accuracy']:.4f}, macro-F1 = {metrics['macro_f1']:.4f}")
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
