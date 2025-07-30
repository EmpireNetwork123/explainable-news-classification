import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_and_save(raw_path, out_dir):
    df = pd.read_csv(raw_path)
    df['clean_text'] = df['text'].apply(clean_text)

    # Optional: map label to name
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    df['label_name'] = df['label'].map(label_map)

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "ag_news_clean_train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "ag_news_clean_val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "ag_news_clean_test.csv"), index=False)

if __name__ == "__main__":
    preprocess_and_save("data/raw/ag_news_train.csv", "data/processed")
