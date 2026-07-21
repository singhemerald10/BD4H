import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'output/confusion_matrix_{model_name}.png')
    plt.close()

def save_classification_report(y_true, y_pred, model_name):
    df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
    df.to_csv(f'output/classification_report_{model_name}.csv')

def save_confidence_histogram(probs, model_name):
    plt.hist(probs, bins=50, alpha=0.7)
    plt.title(f'Confidence Histogram - {model_name}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.savefig(f'output/confidence_histogram_{model_name}.png')
    plt.close()

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 64
    epochs = 30
    learning_rate = 2e-5

    index_df = pd.read_csv("data/processed/embedding_index.csv")
    labels = index_df["LABEL"].values
    embeddings = np.load("data/processed/embeddings.npy")

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(X, y)

    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)

    class ClinicalBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(768, hidden_dim)
            self.out = nn.Linear(hidden_dim, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.out(x)

    model = ClinicalBERTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = torch.tensor([1.0, 4.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    print(f"\nTraining ClinicalBERT Classifier | hidden={hidden_dim}, device={device}")
    best_auroc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_score.extend(probs)
                y_true.extend(yb.numpy())

        auroc = roc_auc_score(y_true, y_score)
        print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f} | AUROC: {auroc:.4f}")
        if auroc > best_auroc:
            best_auroc = auroc

    y_pred = [1 if p >= 0.5 else 0 for p in y_score]
    auprc = average_precision_score(y_true, y_score)
    r_at_p80 = sum((np.array(y_score) >= 0.8) & (np.array(y_true) == 1)) / max(1, sum(np.array(y_score) >= 0.8))

    # 🔁 Updated CSV output format
    result_df = pd.DataFrame({
        "Metric": ["AUROC", "AUPRC", "R@P80%"],
        "Value": [best_auroc, auprc, r_at_p80]
    })
    result_df.to_csv("output/clinicalbert_results.csv", index=False)

    save_confusion_matrix(y_true, y_pred, "ClinicalBERT")
    save_classification_report(y_true, y_pred, "ClinicalBERT")
    save_confidence_histogram(np.array(y_score), "ClinicalBERT")

    print(f"\nFinal ClinicalBERT Results:\nAUROC     : {best_auroc:.4f}\nAUPRC     : {auprc:.4f}\nR@P80%    : {r_at_p80:.4f}")
    print("Cleared CUDA memory after training ClinicalBERT")
