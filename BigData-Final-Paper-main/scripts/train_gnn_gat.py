import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import matplotlib.pyplot as plt

def run():
    print("\nTraining GAT on full Top-K graph")

    # Hyperparameters
    hidden_dim = 32
    heads = 2
    epochs = 200
    input_path = "data/processed"
    output_path = "output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    embeddings = torch.tensor(np.load(f"{input_path}/embeddings.npy")).float().to(device)
    labels = torch.tensor(pd.read_csv(f"{input_path}/embedding_index.csv")["LABEL"].values).long().to(device)
    edge_index = torch.load(f"{input_path}/topk_graph.pt").to(device)

    class GAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(embeddings.shape[1], hidden_dim, heads=heads, dropout=0.6)
            self.lin = nn.Linear(hidden_dim * heads, 2)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.lin(x)
            return x

    model = GAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Train-test split
    idx = torch.arange(embeddings.size(0))
    train_idx, test_idx = train_test_split(idx.cpu(), test_size=0.2, stratify=labels.cpu(), random_state=42)
    train_idx, test_idx = train_idx.to(device), test_idx.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(embeddings, edge_index)
        loss = criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(embeddings, edge_index)
                probs = F.softmax(logits[test_idx], dim=1)[:, 1].detach().cpu()
                preds = probs.numpy()
                true = labels[test_idx].cpu().numpy()
                auroc = roc_auc_score(true, preds)
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | AUROC: {auroc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(embeddings, edge_index)
        probs = F.softmax(logits[test_idx], dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        true = labels[test_idx].cpu().numpy()

        auroc = roc_auc_score(true, probs)
        auprc = average_precision_score(true, probs)

        sorted_preds = sorted(zip(probs, true), key=lambda x: -x[0])
        p_cut = int(len(sorted_preds) * 0.8)
        recall_at_precision_80 = sum([label for _, label in sorted_preds[:p_cut]]) / sum(true) if sum(true) > 0 else 0.0

    print("\nFinal GAT Results:")
    print(f"AUROC     : {auroc:.4f}")
    print(f"AUPRC     : {auprc:.4f}")
    print(f"R@P80%    : {recall_at_precision_80:.4f}")

    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame({
        "Metric": ["AUROC", "AUPRC", "R@P80%"],
        "Value": [auroc, auprc, recall_at_precision_80]
    }).to_csv(f"{output_path}/gat_results.csv", index=False)

    # Confidence histogram
    plt.hist(probs, bins=20, color='royalblue', alpha=0.7)
    plt.title("GAT Prediction Confidence")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_path}/confidence_histogram_gat.png")
    plt.close()

    torch.cuda.empty_cache()
    print("Cleared CUDA memory after training GAT")
