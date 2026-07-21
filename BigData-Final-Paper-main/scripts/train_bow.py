import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

def run():
    print("\nTraining Bag-of-Words | max_features=10000")

    input_path = "data/processed"
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(f"{input_path}/discharge_labeled.csv")
    texts = df["TEXT"].fillna("").values
    labels = df["LABEL"].values

    vectorizer = CountVectorizer(max_features=10000)
    features = vectorizer.fit_transform(texts)

    X = features.toarray()
    y = labels

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)

    sorted_preds = sorted(zip(probs, y_test), key=lambda x: -x[0])
    p_cut = int(len(sorted_preds) * 0.8)
    recall_at_precision_80 = sum([label for _, label in sorted_preds[:p_cut]]) / sum(y_test) if sum(y_test) > 0 else 0.0

    print("\nBag-of-Words Results:")
    print(f"AUROC     : {auroc:.4f}")
    print(f"AUPRC     : {auprc:.4f}")
    print(f"R@P80%    : {recall_at_precision_80:.4f}")

    pd.DataFrame({
        "Metric": ["AUROC", "AUPRC", "R@P80%"],
        "Value": [auroc, auprc, recall_at_precision_80]
    }).to_csv(f"{output_path}/bow_results.csv", index=False)

    plt.hist(probs, bins=20, color='slateblue', alpha=0.7)
    plt.title("Bag-of-Words Prediction Confidence")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_path}/confidence_histogram_bow.png")
    plt.close()
