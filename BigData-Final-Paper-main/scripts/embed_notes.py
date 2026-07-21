import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

def run():
    print("\nEmbedding clinical notes using ClinicalBERT (batched + chunked)")

    # Load labeled discharge summaries
    df = pd.read_csv("data/processed/discharge_labeled.csv").dropna(subset=["TEXT", "HADM_ID"])
    texts = df["TEXT"].tolist()
    labels = df["LABEL"].tolist()
    hadm_ids = df["HADM_ID"].tolist()

    # Load ClinicalBERT
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_embeddings = []
    all_index = []

    for i, text in enumerate(tqdm(texts, desc="Embedding notes")):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = [tokens[j:j+512] for j in range(0, len(tokens), 512)]
        chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

        # ⏩ Batch the chunks (e.g., in groups of 8)
        batch_embeddings = []
        batch_size = 8
        for b in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[b:b+batch_size]
            inputs = tokenizer(batch,
                               return_tensors="pt",
                               padding="max_length",
                               truncation=True,
                               max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # shape: (batch, 768)
                batch_embeddings.extend(cls_embeddings.cpu().numpy())

        if batch_embeddings:
            final_embedding = np.mean(batch_embeddings, axis=0)
        else:
            final_embedding = np.zeros(768)

        all_embeddings.append(final_embedding)
        all_index.append({
            "HADM_ID": hadm_ids[i],
            "LABEL": labels[i]
        })

    # Save
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/embeddings.npy", np.stack(all_embeddings))
    pd.DataFrame(all_index).to_csv("data/processed/embedding_index.csv", index=False)

    print("Saved ClinicalBERT embeddings and index.")
