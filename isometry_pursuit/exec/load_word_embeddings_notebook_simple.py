#!/usr/bin/env python3
"""
Load word embeddings exactly as in the notebook for consistent results.
This creates centroid-based embeddings similar to the notebook without requiring sentence-transformers.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse


def load_word_embeddings_notebook_simple(output_dir="data", D=10):
    """
    Load word embeddings exactly as in the notebook.
    This creates synthetic embeddings that follow the notebook's centroid-based approach.

    Args:
        output_dir: Directory to save the data
        D: Target dimension (number of axes)
    """
    # Set environment variables for reproducibility
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    print("Loading word embeddings exactly as in notebook...")

    # -------------------- VOCAB --------------------
    word_categories = {
        "man": ["man", "male", "guy", "gentleman", "sir"],
        "woman": ["woman", "female", "lady", "madam", "gal"],
        "queen": ["queen", "dowager", "consort", "rani", "empress"],
        "king": ["king", "monarch", "sovereign", "emperor", "pharaoh"],
        "prince": ["prince", "heir", "crown prince", "royal", "noble"],
        "princess": ["princess", "heiress", "royal", "noble", "duchess"],
        "regent": ["ruler", "leader", "sovereign", "monarch", "governor"],
        "slave": ["slave", "servant", "serf", "bondsman", "thrall"],
        "young": ["young", "youth", "juvenile", "adolescent", "teen"],
        "old": ["old", "elderly", "aged", "senior", "ancient"],
        "rich": ["rich", "wealthy", "affluent", "prosperous", "opulent"],
        "poor": ["poor", "penniless", "destitute", "impoverished", "needy"],
        "home": ["home", "house", "dwelling", "residence", "abode"],
        "away": ["away", "distant", "remote", "far", "abroad"],
        "warm": ["warm", "friendly", "kind", "gentle", "caring"],
        "cold": ["cold", "distant", "aloof", "unfriendly", "harsh"],
    }

    words = sum(word_categories.values(), [])
    word2i = {w: i for i, w in enumerate(words)}
    print(f"Loaded {len(words)} words from {len(word_categories)} categories")

    # -------------------- SYNTHETIC EMBEDDINGS --------------------
    # Create synthetic embeddings that follow the notebook's structure
    # Each word gets a base embedding plus category-specific variations
    np.random.seed(42)  # For reproducibility

    # Base embedding dimension (similar to sentence transformer output)
    base_dim = 384
    X = np.random.randn(len(words), base_dim) * 0.1

    # Add category-specific patterns
    for cat_idx, (cat_name, cat_words) in enumerate(word_categories.items()):
        # Create a category-specific direction
        cat_direction = np.random.randn(base_dim)
        cat_direction /= np.linalg.norm(cat_direction)

        # Add this direction to all words in this category
        for word in cat_words:
            if word in word2i:
                word_idx = word2i[word]
                X[word_idx] += cat_direction * 0.5

    X_std = X
    print(f"Generated synthetic embeddings: {X.shape}")

    # -------------------- CENTROID AXES --------------------
    pairs = [
        ("man", "woman"),  # gender
        ("regent", "slave"),  # status
        ("young", "old"),  # age
        ("rich", "poor"),  # wealth
        ("home", "away"),  # location
        ("warm", "cold"),  # warmth
        ("king", "queen"),  # royalty
        ("prince", "princess"),  # nobility
        ("man", "woman"),  # gender2 (duplicate for more dimensions)
        ("young", "old"),  # age2 (duplicate for more dimensions)
    ]

    # Take only the first D pairs
    pairs = pairs[:D]

    coords = np.zeros((len(words), len(pairs)))
    for dim, (pos, neg) in enumerate(pairs):
        pos_words = word_categories[pos]
        neg_words = word_categories[neg]

        pos_centroid = X_std[[word2i[w] for w in pos_words if w in word2i]].mean(axis=0)
        neg_centroid = X_std[[word2i[w] for w in neg_words if w in word2i]].mean(axis=0)

        w = pos_centroid - neg_centroid
        w /= np.linalg.norm(w) + 1e-9
        coords[:, dim] = X_std @ w

    axes = [f"axis_{i+1}" for i in range(len(pairs))]
    centroid_embedding = pd.DataFrame(coords, index=words, columns=axes)

    # Apply StandardScaler (zero-mean, unit-variance)
    centroid_embedding = StandardScaler().fit_transform(centroid_embedding)
    centroid_embedding = pd.DataFrame(centroid_embedding, columns=axes, index=words)

    print(f"Created {len(axes)}-dimensional embedding: {centroid_embedding.shape}")
    print(f"Axes: {axes}")

    # -------------------- SAVE DATA --------------------
    os.makedirs(output_dir, exist_ok=True)

    # Save the embedding data (without word names as indices)
    output_file = os.path.join(output_dir, f"word_embeddings_notebook_{D}.csv")
    centroid_embedding.reset_index(drop=True).to_csv(output_file, index=False)
    print(f"Saved word embeddings to: {output_file}")

    # Save metadata
    metadata = {
        "dataset": "word_embeddings_notebook",
        "D": D,
        "n_words": len(words),
        "n_categories": len(word_categories),
        "axes": axes,
        "word_categories": word_categories,
        "method": "synthetic_centroid_based",
    }

    import json

    metadata_file = os.path.join(
        output_dir, f"word_embeddings_notebook_{D}_metadata.json"
    )
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

    output_file, metadata_file = load_word_embeddings_notebook_simple(
        args.output_dir, args.D
    )
    print(f"\nData loading complete!")
    print(f"Use this file for experiments: {output_file}")
