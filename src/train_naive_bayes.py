"""TF-IDF + Multinomial Naive Bayes baseline.

The smoothing parameter alpha is selected on the validation split; the
selected model is then refit on train + validation and scored once on the
official test split.
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src import config
from src.utils import compute_metrics, plot_confusion_matrix, save_json

MODEL_PATH = config.MODELS_DIR / "naive_bayes_pipeline.joblib"


def build_pipeline(alpha):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("nb", MultinomialNB(alpha=alpha)),
    ])


def load_split(name):
    df = pd.read_csv(config.PROCESSED_DIR / f"ag_news_clean_{name}.csv")
    return df["clean_text"].fillna(""), df["label"]


def select_alpha(X_train, y_train, X_val, y_val):
    scores = {}
    for alpha in config.NB_ALPHAS:
        pipe = build_pipeline(alpha).fit(X_train, y_train)
        scores[alpha] = accuracy_score(y_val, pipe.predict(X_val))
        print(f"alpha={alpha:<5} val_accuracy={scores[alpha]:.4f}")
    return max(scores, key=scores.get), scores


def main():
    config.ensure_dirs()
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    best_alpha, val_scores = select_alpha(X_train, y_train, X_val, y_val)
    print(f"Selected alpha = {best_alpha}")

    model = build_pipeline(best_alpha).fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    joblib.dump(model, MODEL_PATH)

    metrics = compute_metrics(y_test, model.predict(X_test))
    metrics.update({
        "model": "TF-IDF (1-2 grams) + Multinomial Naive Bayes",
        "alpha": best_alpha,
        "validation_accuracy_by_alpha": {str(a): s for a, s in val_scores.items()},
    })
    save_json(metrics, config.METRICS_DIR / "naive_bayes_metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"], "Confusion Matrix: TF-IDF + Naive Bayes",
        config.FIGURES_DIR / "nb_confusion_matrix.png",
    )
    print(f"Test accuracy = {metrics['accuracy']:.4f}, macro-F1 = {metrics['macro_f1']:.4f}")
    return model, metrics


if __name__ == "__main__":
    main()
