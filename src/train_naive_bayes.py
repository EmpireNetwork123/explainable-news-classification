import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_naive_bayes(data_path):
    df = pd.read_csv(data_path)
    X = df['clean_text']
    y = df['label']

    tfidf = TfidfVectorizer()
    X_vec = tfidf.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    joblib.dump(model, "results/naive_bayes_model.pkl")
    joblib.dump(tfidf, "results/tfidf_vectorizer.pkl")

    return model, tfidf

def evaluate(model, tfidf, test_path):
    df = pd.read_csv(test_path)
    X_test = tfidf.transform(df['clean_text'])
    y_test = df['label']
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    model, tfidf = train_naive_bayes("data/processed/ag_news_clean_train.csv")
    evaluate(model, tfidf, "data/processed/ag_news_clean_test.csv")
