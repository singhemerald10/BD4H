import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def tokenize(text):
    return text.lower().split()

class DischargeDataset(Dataset):
    def __init__(self, texts, labels, vocab, seq_len=300):
        self.vocab = vocab
        self.texts = [self.pad_or_truncate(vocab(tokenize(t)), seq_len) for t in texts]
        self.labels = labels

    def pad_or_truncate(self, tokens, length):
        return tokens[:length] + [0] * max(0, length - len(tokens))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

def run():
    print("\nTraining BiLSTM | vocab=10000, hidden_dim=128, seq_len=300")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_path = "data/processed"
    output_path = "output"
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(f"{input_path}/discharge_labeled.csv")
    texts = df["TEXT"].fillna("").tolist()
    labels = df["LABEL"].tolist()

    split = int(0.8 * len(texts))
    train_texts, test_texts = texts[:split], texts[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    def yield_tokens(data):
        for text in data:
            yield tokenize(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_texts), specials=["<pad>"], max_tokens=10000)
    vocab.set_default_index(vocab["<pad>"])

    train_dataset = DischargeDataset(train_texts, train_labels, vocab)
    test_dataset = DischargeDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    class BiLSTM(nn.Module):
        def __init__(self, vocab_size, hidden_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
            self.lstm = nn.LSTM(128, hidden_dim, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden_dim * 2, 2)

        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.lstm(x)
            x = x.mean(dim=1)
            return self.fc(x)

    model = BiLSTM(len(vocab), 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            out = model(x_batch)
            prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            probs.extend(prob)
            targets.extend(y_batch.numpy())

    auroc = roc_auc_score(targets, probs)
    auprc = average_precision_score(targets, probs)
    sorted_preds = sorted(zip(probs, targets), key=lambda x: -x[0])
    p_cut = int(len(sorted_preds) * 0.8)
    recall_at_precision_80 = sum([label for _, label in sorted_preds[:p_cut]]) / sum(targets) if sum(targets) > 0 else 0.0

    print("\nBiLSTM Results:")
    print(f"AUROC     : {auroc:.4f}")
    print(f"AUPRC     : {auprc:.4f}")
    print(f"R@P80%    : {recall_at_precision_80:.4f}")

    pd.DataFrame({
        "Metric": ["AUROC", "AUPRC", "R@P80%"],
        "Value": [auroc, auprc, recall_at_precision_80]
    }).to_csv(f"{output_path}/lstm_results.csv", index=False)

    plt.hist(probs, bins=20, color='darkorange', alpha=0.7)
    plt.title("BiLSTM Prediction Confidence")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_path}/confidence_histogram_lstm.png")
    plt.close()

    torch.cuda.empty_cache()
    print("Cleared CUDA memory after training BiLSTM")
