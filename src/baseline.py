import pandas as pd
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

# ── Column names for LIAR TSV ──────────────────────────────────────────────
COLUMNS = [
    'id', 'label', 'statement', 'subject', 'speaker',
    'job_title', 'state_info', 'party_affiliation',
    'barely_true_count', 'false_count', 'half_true_count',
    'mostly_true_count', 'pants_fire_count', 'context'
]

FAKE_LABELS = {'false', 'barely-true', 'pants-fire'}

def load_data(path):
    df = pd.read_csv(path, sep='\t', header=None, names=COLUMNS)
    df['binary_label'] = df['label'].apply(
        lambda x: 1 if x in FAKE_LABELS else 0
    )
    return df

def get_metrics(y_true, y_pred):
    return {
        'accuracy'  : round(accuracy_score(y_true, y_pred), 4),
        'precision' : round(precision_score(y_true, y_pred, average='macro'), 4),
        'recall'    : round(recall_score(y_true, y_pred, average='macro'), 4),
        'f1_macro'  : round(f1_score(y_true, y_pred, average='macro'), 4),
    }

def main():
    print("Loading LIAR dataset...")
    df_train = load_data('data/raw/train.tsv')
    df_test  = load_data('data/raw/test.tsv')

    X_train = df_train['statement'].tolist()
    X_test  = df_test['statement'].tolist()
    y_train = df_train['binary_label'].tolist()
    y_test  = df_test['binary_label'].tolist()

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    results = {}

    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    lr_preds = lr.predict(X_test_tfidf)
    results['logistic_regression'] = get_metrics(y_test, lr_preds)
    print("LR Results:", results['logistic_regression'])

    # Naive Bayes
    print("Training Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    nb_preds = nb.predict(X_test_tfidf)
    results['naive_bayes'] = get_metrics(y_test, nb_preds)
    print("NB Results:", results['naive_bayes'])

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✅ Baseline results saved to results/baseline_results.json")

if __name__ == '__main__':
    main()
