import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

def load_and_prepare_data(train_path, val_path, tokenizer):
    # Load CSVs
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Keep only necessary columns
    train_ds = Dataset.from_pandas(train_df[['clean_text', 'label']])
    val_ds = Dataset.from_pandas(val_df[['clean_text', 'label']])

    # Tokenization function
    def tokenize(batch):
        return tokenizer(batch['clean_text'], padding=True, truncation=True)

    # Apply tokenization
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    return train_ds, val_ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)

def main():
    # Load model and tokenizer
    model_dir = r"C:\Users\Ericmanny\Desktop\bert_base_uncased"
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=4)

    # Load data
    train_path = "data/processed/ag_news_clean_train.csv"
    val_path = "data/processed/ag_news_clean_val.csv"
    train_ds, val_ds = load_and_prepare_data(train_path, val_path, tokenizer)

    # Configurations
    training_args = TrainingArguments(
        output_dir="results/bert_news_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",  # Disable WandB logging unless used
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    # Train & Save
    trainer.train()
    model.save_pretrained("results/bert_final_model")
    tokenizer.save_pretrained("results/bert_final_model")

    print("✅ BERT training complete!")

if __name__ == "__main__":
    main()
