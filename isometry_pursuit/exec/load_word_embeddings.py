#!/usr/bin/env python3
"""
Load word embeddings exactly as in the notebook - no innovation, just copy.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import argparse


def load_word_embeddings_notebook(output_dir="data", D=10):
    """
    Load word embeddings exactly as in the notebook.
    """
    # Set environment variables for reproducibility
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    print("Loading word embeddings exactly as in notebook...")

    # -------------------- VOCAB (exactly from notebook) --------------------
    word_categories = {
        "man": ["man", "male", "guy", "gentleman", "sir"],
        "woman": ["woman", "female", "lady", "madam", "gal"],
        "regent": ["regent", "sovereign", "monarch", "emperor", "pharaoh"],
        "slave": ["slave", "wench"],
        "young": ["young", "kid"],
        "old": ["old", "elderly"],
        "warrior": ["warrior", "soldier", "knight", "fighter", "mercenary"],
        "scholar": ["scholar", "academic", "professor", "intellectual", "scientist"],
        "rich": ["rich", "wealthy", "affluent", "millionaire", "tycoon"],
        "poor": ["poor", "impoverished", "broke", "destitute", "beggar"],
        "urban": ["urban", "city", "metropolitan", "cosmopolitan", "downtown"],
        "rural": ["rural", "village", "countryside", "agrarian", "pastoral"],
        "warm": ["warm", "friendly", "kind", "loving", "affectionate"],
        "cold": ["cold", "aloof", "harsh", "distant", "indifferent"],
    }

    words = sum(word_categories.values(), [])
    word2i = {w: i for i, w in enumerate(words)}
    print(f"Loaded {len(words)} words from {len(word_categories)} categories")

    # -------------------- EMBEDDINGS (exactly from notebook) --------------------
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(words, convert_to_tensor=True).cpu().numpy()
    X_std = X
    print(f"Generated embeddings: {X.shape}")

    # -------------------- CENTROID AXES (exactly from notebook) --------------------
    pairs = [
        ("man", "woman"),
        ("regent", "slave"),
        ("young", "old"),
        ("warrior", "scholar"),
        ("rich", "poor"),
        ("urban", "rural"),
        ("warm", "cold"),
    ]

    # Take only the first D pairs
    pairs = pairs[:D]

    coords = np.zeros((len(words), len(pairs)))
    for dim, (cls_pos, cls_neg) in enumerate(pairs):
        idx_pos = [word2i[w] for w in word_categories[cls_pos]]
        idx_neg = [word2i[w] for w in word_categories[cls_neg]]

        mu_pos = X_std[idx_pos].mean(axis=0)
        mu_neg = X_std[idx_neg].mean(axis=0)
        w = mu_pos - mu_neg
        w /= np.linalg.norm(w) + 1e-9
        coords[:, dim] = X_std @ w

    axes = ["gender", "status", "age", "domain", "wealth", "location", "warmth"]
    axes = axes[:D]  # Take only the first D axes
    centroid_embedding = pd.DataFrame(coords, index=words, columns=axes)

    # Apply StandardScaler (exactly from notebook)
    centroid_embedding = StandardScaler().fit_transform(centroid_embedding)
    centroid_embedding = pd.DataFrame(centroid_embedding, columns=axes, index=words)

    # -------------------- CATEGORY LABEL (exactly from notebook) --------------------
    category_of = {w: cat for cat, vocab in word_categories.items() for w in vocab}
    centroid_embedding["category"] = centroid_embedding.index.map(category_of).astype(
        "category"
    )

    # Final embeds variable (exactly from notebook)
    embeds = centroid_embedding[axes]

    print(f"Created {len(axes)}-dimensional embedding: {embeds.shape}")
    print(f"Axes: {axes}")

    # -------------------- SAVE DATA --------------------
    os.makedirs(output_dir, exist_ok=True)

    # Save the embedding data (without word names as indices for compatibility)
    output_file = os.path.join(output_dir, f"word_embeddings_{D}.csv")
    embeds.reset_index(drop=True).to_csv(output_file, index=False)
    print(f"Saved word embeddings to: {output_file}")

    # Save metadata
    metadata = {
        "dataset": "word_embeddings",
        "D": D,
        "n_words": len(words),
        "n_categories": len(word_categories),
        "axes": axes,
        "word_categories": word_categories,
        "model": "all-MiniLM-L6-v2",
        "method": "notebook_replication",
    }

    import json

    metadata_file = os.path.join(output_dir, f"word_embeddings_{D}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")

    return output_file, metadata_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load word embeddings exactly as in notebook"
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--D", type=int, default=10, help="Target dimension")

    args = parser.parse_args()

    output_file, metadata_file = load_word_embeddings_notebook(args.output_dir, args.D)
    print(f"\nData loading complete!")
    print(f"Use this file for experiments: {output_file}")
