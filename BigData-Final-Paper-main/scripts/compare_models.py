import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def run():
    print("\nComparing model results...")

    model_files = {
        "ClinicalBERT": "output/clinicalbert_results.csv",
        "GCN": "output/gcn_results.csv",
        "GAT": "output/gat_results.csv",
        "BoW": "output/bow_results.csv",
        "BiLSTM": "output/lstm_results.csv",
    }

    metrics = ["AUROC", "AUPRC", "R@P80%"]
    model_names = list(model_files.keys())
    scores_by_metric = {metric: [] for metric in metrics}

    for model in model_names:
        file_path = model_files[model]
        if not os.path.exists(file_path):
            print(f"WARNING: Missing result file: {file_path}")
            for metric in metrics:
                scores_by_metric[metric].append(0.0)
            continue

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df["Metric"] = df["Metric"].str.strip()

        print(f"\n{model} Results:")
        for metric in metrics:
            try:
                value = df.loc[df["Metric"] == metric, "Value"].values[0]
            except IndexError:
                value = 0.0
            print(f"  {metric}: {value:.4f}")
            scores_by_metric[metric].append(value)

    # Plot grouped bar chart
    x = np.arange(len(metrics))
    width = 0.15

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(model_names):
        model_scores = [scores_by_metric[metric][i] for metric in metrics]
        plt.bar(x + i * width - width * 2, model_scores, width=width, label=model)

    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Metric-wise Model Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/model_comparison.png")

    print("\nModel comparison chart saved to: output/model_comparison.png")
