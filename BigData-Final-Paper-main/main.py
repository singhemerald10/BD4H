import os
import gc
import torch
import warnings

from scripts import compare_models

warnings.filterwarnings("ignore")

from scripts.extract_notes import run as extract_notes_run
from scripts.embed_notes import run as embed_notes_run
from scripts.build_graph import run as build_graph_run

from scripts.train_bow import run as bow_train
from scripts.train_clinical_bert import run as clinicalbert_train
from scripts.train_gnn_gcn import run as gcn_train
from scripts.train_gnn_gat import run as gat_train
from scripts.train_lstm import run as lstm_train


def safe_run(label, trainer_fn):
    print(f"\n{'=' * 50}\n Starting {label}")
    trainer_fn()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f" Cleared CUDA memory after {label}")


def main():
    print("\n DeepNote-GNN Reproduction Pipeline")

    safe_run("Step 1: Extract Notes & Label", extract_notes_run)
    safe_run("Step 2: ClinicalBERT Embeddings", embed_notes_run)
    safe_run("Step 3: Graph Construction", lambda: build_graph_run(k=134))

    # # Models
    safe_run("BoW Baseline", bow_train)
    safe_run("BiLSTM Baseline", lstm_train)
    safe_run("ClinicalBERT Classifier", clinicalbert_train)
    safe_run("GCN Patient Network", gcn_train)
    safe_run("GAT Patient Network", gat_train)
    compare_models.run()

    print("\n All stages complete. Check `output/` for results.")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    main()
