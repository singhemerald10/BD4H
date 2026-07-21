import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def run(k=134):
    print(f"\nBuilding patient graph with Top-{k} cosine similarity (batched, GPU-safe)")

    input_path = "data/processed"
    emb_file = os.path.join(input_path, "embeddings.npy")
    idx_file = os.path.join(input_path, "embedding_index.csv")
    out_file = os.path.join(input_path, "topk_graph.pt")

    # Load data
    embeddings = np.load(emb_file)
    index_df = pd.read_csv(idx_file)

    num_nodes = embeddings.shape[0]
    edge_list = []

    batch_size = 256
    for i in tqdm(range(0, num_nodes, batch_size), desc="🔍 Computing edges"):
        end = min(i + batch_size, num_nodes)
        sim = cosine_similarity(embeddings[i:end], embeddings)
        top_k_indices = np.argpartition(-sim, kth=k, axis=1)[:, :k]

        for j in range(end - i):
            src = i + j
            neighbors = top_k_indices[j]
            for dst in neighbors:
                if src != dst:
                    edge_list.append((src, dst))

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    print(f"Saved Top-{k} Graph: {edge_index.size(1):,} edges from {num_nodes} nodes")

    os.makedirs(input_path, exist_ok=True)
    torch.save(edge_index, out_file)
