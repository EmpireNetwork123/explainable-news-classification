import pandas as pd
import joblib
import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

def explain_instance(example_id=0):
    df = pd.read_csv("data/processed/ag_news_clean_test.csv")
    class_names = ["World", "Sports", "Business", "Sci/Tech"]

    model = joblib.load("results/naive_bayes_model.pkl")
    vectorizer = joblib.load("results/tfidf_vectorizer.pkl")

    pipeline = lambda x: model.predict_proba(vectorizer.transform(x))

    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(df.loc[example_id, 'clean_text'], pipeline, num_features=10)
    
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig("results/lime_explanation.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    explain_instance(0)
