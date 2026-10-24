"""Local explanations with LIME for either trained model.

LIME perturbs the input article by removing words, queries the classifier on
the perturbed copies and fits a sparse linear surrogate. We explain the class
that the model actually predicts for the selected article.

Example:
    python -m src.explain_model --model nb --index 0
    python -m src.explain_model --model bert --index 0
"""
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

from src import config


def nb_predictor():
    model = joblib.load(config.MODELS_DIR / "naive_bayes_pipeline.joblib")
    return model.predict_proba, "clean_text"


def bert_predictor(model_dir=config.MODELS_DIR / "bert_final", batch_size=64):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    def predict_proba(texts):
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(list(texts[i:i + batch_size]), truncation=True,
                                max_length=config.BERT_MAX_LENGTH, padding=True,
                                return_tensors="pt").to(device)
                probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
        return np.vstack(probs)

    return predict_proba, "text"


def explain(model="nb", index=0, num_features=10, num_samples=5000):
    predict_proba, column = nb_predictor() if model == "nb" else bert_predictor()
    df = pd.read_csv(config.PROCESSED_DIR / "ag_news_clean_test.csv")
    article = str(df.loc[index, column])
    true_label = config.LABEL_NAMES[int(df.loc[index, "label"])]

    explainer = LimeTextExplainer(class_names=config.LABEL_NAMES, random_state=config.SEED)
    exp = explainer.explain_instance(
        article, predict_proba, num_features=num_features,
        num_samples=num_samples, top_labels=1,
    )
    pred = exp.top_labels[0]
    pred_label = config.LABEL_NAMES[pred]

    fig = exp.as_pyplot_figure(label=pred)
    fig.set_size_inches(7, 4.5)
    plt.title(f"LIME ({model.upper()}): predicted {pred_label}, true {true_label}")
    plt.tight_layout()
    stem = f"lime_{model}_test{index}"
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    exp.save_to_file(str(config.FIGURES_DIR / f"{stem}.html"))

    print(f"Article: {article}")
    print(f"True label: {true_label} | predicted: {pred_label} (p = {exp.predict_proba[pred]:.3f})")
    for word, weight in exp.as_list(label=pred):
        print(f"  {word:<20} {weight:+.4f}")
    return exp


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["nb", "bert"], default="nb")
    p.add_argument("--index", type=int, default=0, help="Row of the test split to explain.")
    p.add_argument("--num-features", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=5000)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    explain(a.model, a.index, a.num_features, a.num_samples)
