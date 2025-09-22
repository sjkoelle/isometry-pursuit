#!/usr/bin/env python3
"""
Word Embeddings Experiment
This script runs isometry pursuit experiments on word embeddings.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import isometry_pursuit modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from isometry_pursuit.experiment import run_resampling_experiment


def load_word_embeddings(output_dir="data", D=10, standardize=True):
    """
    Load and preprocess word embeddings.

    Parameters:
    -----------
    output_dir : str
        Directory to save the processed data
    D : int
        Target dimension for PCA
    standardize : bool
        Whether to standardize the data

    Returns:
    --------
    data : pandas.DataFrame
        Processed word embeddings data
    D : int
        Number of features (dimension)
    P : int
        Number of samples
    """

    print("Loading word embeddings...")

    # Set environment variables for single-threaded execution
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Load a lightweight sentence embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Define word categories
    word_categories = {
        "man": [
            "man",
            "male",
            "guy",
            "gentleman",
            "sir",
        ],
        "woman": [
            "woman",
            "female",
            "lady",
            "madam",
            "gal",
        ],
        "regent": [  # high‑status / ruler
            "regent",
            "sovereign",
            "monarch",
            "emperor",
            "pharaoh",
        ],
        "slave": [  # low‑status / servitude
            "slave",
            "wench",
        ],
        "young": [
            "young",
            "kid",
        ],
        "old": [
            "old",
            "elder",
        ],
    }

    # Flatten word list
    words = []
    for category, word_list in word_categories.items():
        words.extend(word_list)

    print(f"Processing {len(words)} words from {len(word_categories)} categories")

    # Get embeddings
    embeddings = model.encode(words, convert_to_tensor=True).cpu()

    # Convert to DataFrame
    features = embeddings.t()
    print(f"Original embedding shape: {features.shape}")

    if standardize:
        print("Standardizing embeddings...")
        scaler = StandardScaler()
        features_standardized = pd.DataFrame(
            data=scaler.fit_transform(features),
            columns=[f"dim_{i}" for i in range(features.shape[1])],
        )

        # Verify standardization
        norms = np.linalg.norm(features_standardized, axis=0)
        print(f"Column norms after standardization: {norms[:5]}...")

        data = features_standardized
        data_name = "word_embeddings_standardized"
    else:
        data = pd.DataFrame(
            features, columns=[f"dim_{i}" for i in range(features.shape[1])]
        )
        data_name = "word_embeddings"

    # Apply PCA to reduce dimensionality
    print(f"Applying PCA to reduce to {D} dimensions...")
    pca = PCA(n_components=D)
    embeds = pd.DataFrame(
        pca.fit_transform(data.transpose()), columns=[f"pc_{i}" for i in range(D)]
    )

    print(f"Final embedding shape: {embeds.shape}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:5]}...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save data
    output_file = os.path.join(output_dir, f"{data_name}_{D}.csv")
    embeds.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    # Save metadata
    metadata = {
        "dataset_name": "word_embeddings",
        "standardized": standardize,
        "n_words": len(words),
        "n_categories": len(word_categories),
        "original_dim": features.shape[1],
        "pca_dim": D,
        "words": words,
        "categories": word_categories,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "data_file": output_file,
    }

    metadata_file = os.path.join(output_dir, f"{data_name}_{D}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")

    return embeds, D, len(words)


def run_word_embeddings_experiment(
    data,
    D,
    power=1.0,
    frac=0.5,
    R=25,
    compute_brute=False,
    log_convergence=False,
    output_dir="results",
):
    """
    Run isometry pursuit experiment on word embeddings.

    Parameters:
    -----------
    data : pandas.DataFrame
        Input word embeddings data
    D : int
        Target dimension
    power : float
        Power parameter for transformation
    frac : float
        Fraction of data to use in each resampling
    R : int
        Number of resampling runs
    compute_brute : bool
        Whether to compute brute force solutions
    log_convergence : bool
        Whether to log convergence information
    output_dir : str
        Directory to save results

    Returns:
    --------
    results_df : pandas.DataFrame
        Results from the experiment
    """

    print(f"Running word embeddings experiment...")
    print(f"Parameters: D={D}, power={power}, frac={frac}, R={R}")
    print(f"Data shape: {data.shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run the experiment
    if log_convergence:
        (
            results_df,
            co_occurrence_matrix,
            greedy_co_occurrence_matrix,
            convergence_logs,
        ) = run_resampling_experiment(
            data,
            D,
            frac=frac,
            R=R,
            power=power,
            compute_brute=compute_brute,
            log_convergence=log_convergence,
        )
    else:
        results_df, co_occurrence_matrix, greedy_co_occurrence_matrix = (
            run_resampling_experiment(
                data,
                D,
                frac=frac,
                R=R,
                power=power,
                compute_brute=compute_brute,
                log_convergence=log_convergence,
            )
        )
        convergence_logs = None

    # Save results
    results_file = os.path.join(output_dir, f"word_embeddings_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Save co-occurrence matrices
    co_occurrence_file = os.path.join(
        output_dir, f"word_embeddings_standardized_{power}_{D}_co_occurrence.pkl"
    )
    import pickle

    with open(co_occurrence_file, "wb") as f:
        pickle.dump(
            {
                "co_occurrence_matrix": co_occurrence_matrix,
                "greedy_co_occurrence_matrix": greedy_co_occurrence_matrix,
            },
            f,
        )
    print(f"Co-occurrence matrices saved to {co_occurrence_file}")

    # Save convergence logs if available
    if convergence_logs is not None:
        convergence_file = os.path.join(
            output_dir, f"word_embeddings_standardized_{power}_{D}_convergence.pkl"
        )
        with open(convergence_file, "wb") as f:
            pickle.dump(convergence_logs, f)
        print(f"Convergence logs saved to {convergence_file}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run word embeddings experiment")
    parser.add_argument(
        "--data-dir", default="data", help="Directory containing word embeddings data"
    )
    parser.add_argument("--D", type=int, default=10, help="Target dimension")
    parser.add_argument("--power", type=float, default=1.0, help="Power parameter")
    parser.add_argument(
        "--frac", type=float, default=0.5, help="Fraction of data to use"
    )
    parser.add_argument("--R", type=int, default=25, help="Number of resampling runs")
    parser.add_argument(
        "--compute-brute", action="store_true", help="Compute brute force solutions"
    )
    parser.add_argument(
        "--log-convergence", action="store_true", help="Log convergence information"
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument(
        "--load-data", action="store_true", help="Load and preprocess data first"
    )

    args = parser.parse_args()

    # Load data if requested
    if args.load_data:
        data, D, P = load_word_embeddings(
            output_dir=args.data_dir, D=args.D, standardize=True
        )
    else:
        # Try to load existing data
        data_file = os.path.join(
            args.data_dir, f"word_embeddings_standardized_{args.D}.csv"
        )
        if not os.path.exists(data_file):
            print(f"Data file not found: {data_file}")
            print("Use --load-data to generate data first")
            return

        data = pd.read_csv(data_file)
        D = args.D
        P = data.shape[0]
        print(f"Loaded existing data: {P} samples, {D} features")

    # Run experiment
    results_df = run_word_embeddings_experiment(
        data=data,
        D=D,
        power=args.power,
        frac=args.frac,
        R=args.R,
        compute_brute=args.compute_brute,
        log_convergence=args.log_convergence,
        output_dir=args.output_dir,
    )

    print("Word embeddings experiment completed successfully!")
    print(f"Results shape: {results_df.shape}")


if __name__ == "__main__":
    main()
