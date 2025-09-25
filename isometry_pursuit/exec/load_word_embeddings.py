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


def load_word_embeddings_notebook(output_dir="data", n_replicates=25, frac=0.5):
    """
    Load word embeddings exactly as in the notebook with resampling.

    Parameters:
    -----------
    output_dir : str
        Directory to save the processed data
    standardize : bool
        Whether to standardize the data
    n_replicates : int
        Number of resampling replicates to generate
    frac : float
        Fraction of data to use in each resampling

    Returns:
    --------
    data_tensor : numpy.ndarray
        Resampled data tensor of shape (D, n_samples_per_replicate, n_replicates)
    data : pandas.DataFrame
        Original processed word embeddings data
    D : int
        Number of features (dimension)
    P : int
        Number of samples
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

    # Use all available pairs

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
    centroid_embedding = pd.DataFrame(coords, index=words, columns=axes)

    # Apply StandardScaler (exactly from notebook)
    # centroid_embedding = StandardScaler().fit_transform(centroid_embedding)
    # centroid_embedding = pd.DataFrame(centroid_embedding, columns=axes, index=words)

    # -------------------- CATEGORY LABEL (exactly from notebook) --------------------
    category_of = {w: cat for cat, vocab in word_categories.items() for w in vocab}
    centroid_embedding["category"] = centroid_embedding.index.map(category_of).astype(
        "category"
    )

    # Final embeds variable (exactly from notebook)
    embeds = centroid_embedding[axes]

    print(f"Created {len(axes)}-dimensional embedding: {embeds.shape}")
    print(f"Axes: {axes}")

    # Get basic info
    P_words, D_words = embeds.shape
    print(f"Word embeddings dataset: {P_words} samples, {D_words} features")

    print("Standardizing data...")
    scaler = StandardScaler()
    embeds_standardized = pd.DataFrame(
        data=scaler.fit_transform(embeds), columns=embeds.columns
    )

    # Verify standardization (should be sqrt(P))
    norms = np.linalg.norm(embeds_standardized, axis=0)
    print(f"Column norms after standardization: {norms}")

    data = embeds_standardized
    data_name = "word_embeddings_standardized"

    # -------------------- SAVE DATA --------------------
    os.makedirs(output_dir, exist_ok=True)

    # Save the embedding data (without word names as indices for compatibility)
    output_file = os.path.join(output_dir, f"{data_name}.csv")
    data.reset_index(drop=True).to_csv(output_file, index=False)
    print(f"Saved word embeddings to: {output_file}")

    # Save metadata
    metadata = {
        "dataset": "word_embeddings",
        "D": D_words,
        "n_words": len(words),
        "n_categories": len(word_categories),
        "axes": axes,
        "word_categories": word_categories,
        "model": "all-MiniLM-L6-v2",
        "method": "notebook_replication",
        "data_file": output_file,
    }

    import json

    metadata_file = os.path.join(output_dir, f"{data_name}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")

    # Generate resampled data tensor
    print(f"Generating {n_replicates} resampled replicates with frac={frac}...")
    n_samples_per_replicate = int(P_words * frac)
    data_tensor = np.zeros((D_words, n_samples_per_replicate, n_replicates))

    for i in range(n_replicates):
        np.random.seed(i)
        random_indices = np.random.choice(
            range(P_words), n_samples_per_replicate, replace=False
        )
        data_tensor[:, :, i] = data.to_numpy()[random_indices].T

    print(f"Generated data tensor shape: {data_tensor.shape}")

    # Save resampled data tensor
    tensor_file = os.path.join(output_dir, f"{data_name}_resampled_tensor.npy")
    np.save(tensor_file, data_tensor)
    print(f"Resampled data tensor saved to {tensor_file}")

    return data_tensor, data, D_words, P_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load word embeddings exactly as in notebook with resampling"
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument(
        "--n-replicates", type=int, default=25, help="Number of resampling replicates"
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.5,
        help="Fraction of data to use in each resampling",
    )

    args = parser.parse_args()

    data_tensor, data, D, P = load_word_embeddings_notebook(
        output_dir=args.output_dir,
        n_replicates=args.n_replicates,
        frac=args.frac,
    )
    print(f"\nData loading complete!")
    print(f"Generated resampled tensor: {data_tensor.shape}")
    print(f"Original data shape: {data.shape}")
    print(
        f"Use the tensor file for experiments: {args.output_dir}/word_embeddings_standardized_resampled_tensor.npy"
    )
