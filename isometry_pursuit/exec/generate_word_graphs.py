#!/usr/bin/env python3
"""
Generate co-occurrence network graphs for word embeddings experiment.
Based on the logic from experiments/word_embeddings.ipynb.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description="Generate word co-occurrence graphs")
    parser.add_argument(
        "--co-occurrence-file",
        required=True,
        help="Path to co-occurrence matrices pickle file",
    )
    parser.add_argument(
        "--output-dir", default="figures", help="Directory to save generated graphs"
    )

    args = parser.parse_args()

    # Load co-occurrence matrices
    if not os.path.exists(args.co_occurrence_file):
        print(f"Error: Co-occurrence file not found: {args.co_occurrence_file}")
        sys.exit(1)

    with open(args.co_occurrence_file, "rb") as f:
        co_occurrence_data = pickle.load(f)

    # Extract matrices (structure depends on how co-occurrence data was saved)
    if isinstance(co_occurrence_data, dict):
        # Dictionary format: {'co_occurrence_matrix': ..., 'greedy_co_occurrence_matrix': ...}
        ip_co_occurrence = co_occurrence_data["co_occurrence_matrix"]
        greedy_co_occurrence = co_occurrence_data["greedy_co_occurrence_matrix"]
    else:
        # List/tuple format from run_resampling_experiment
        ip_co_occurrence = co_occurrence_data[1]  # Two-stage IP co-occurrence
        greedy_co_occurrence = co_occurrence_data[2]  # Greedy co-occurrence

    # Load word categories and vocabulary from the metadata file
    # Extract base dataset name (e.g., word_embeddings_standardized_1.0_10 -> word_embeddings_standardized_10)
    base_name = os.path.basename(args.co_occurrence_file)
    dataset_name = base_name.replace("_co_occurrence.pkl", "").replace("_1.0_10", "_10")
    metadata_file = f"../../data/{dataset_name}_metadata.json"

    try:
        import json

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        words = metadata["words"]
        word_categories = metadata["categories"]
        category_of = {w: cat for cat, vocab in word_categories.items() for w in vocab}

        print(
            f"Loaded vocabulary: {len(words)} words from {len(word_categories)} categories"
        )

    except FileNotFoundError:
        print(f"Warning: Could not find metadata file {metadata_file}")
        print("Using default vocabulary from notebook")

        # Fallback to original vocabulary
        word_categories = {
            "man": ["man", "male", "guy", "gentleman", "sir"],
            "woman": ["woman", "female", "lady", "madam", "gal"],
            "regent": ["regent", "sovereign", "monarch", "emperor", "pharaoh"],
            "slave": ["slave", "wench"],
            "young": ["young", "kid"],
            "old": ["old", "elderly"],
        }

        words = sum(word_categories.values(), [])
        category_of = {w: cat for cat, vocab in word_categories.items() for w in vocab}

    # Color mapping by semantic axis (based on actual categories in data)
    axis_of_category = {
        "man": "gender",
        "woman": "gender",
        "regent": "status",
        "slave": "status",
        "young": "age",
        "old": "age",
    }

    axis_palette = {
        "gender": "tab:purple",
        "status": "tab:orange",
        "age": "tab:green",
    }

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate graphs
    generate_graph(
        ip_co_occurrence,
        words,
        category_of,
        axis_of_category,
        axis_palette,
        "Two-stage isometry pursuit replicate",
        os.path.join(args.output_dir, "ip_graph.png"),
    )

    generate_graph(
        greedy_co_occurrence,
        words,
        category_of,
        axis_of_category,
        axis_palette,
        "Greedy replicate",
        os.path.join(args.output_dir, "greedy_graph.png"),
    )

    print(f"Generated co-occurrence graphs in {args.output_dir}/")


def generate_graph(
    co_occurrence_matrix,
    words,
    category_of,
    axis_of_category,
    axis_palette,
    title,
    output_path,
):
    """Generate a single co-occurrence network graph."""

    # Build NetworkX graph
    G = nx.Graph()
    for i, word in enumerate(words):
        G.add_node(word)

    # Add edges (threshold >= 5 co-occurrences)
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            w = co_occurrence_matrix[i, j]
            if w >= 5:
                G.add_edge(words[i], words[j], weight=w)

    # Focus on largest connected component
    if len(G.edges) == 0:
        print(f"Warning: No edges found for {title} - using lower threshold")
        # Try with threshold >= 1 if no edges with >= 5
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w = co_occurrence_matrix[i, j]
                if w >= 1:
                    G.add_edge(words[i], words[j], weight=w)

    if len(G.edges) == 0:
        print(f"Warning: Still no edges found for {title} - creating minimal graph")
        # If still no edges, create a minimal graph with isolated nodes
        G_sub = G
    else:
        try:
            G_sub = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        except ValueError:
            # If no connected components, use the full graph
            G_sub = G

    # Node sizes based on total co-occurrence strength
    node_strength = co_occurrence_matrix.sum(axis=1)
    s_min, s_max = 400, 3000
    if node_strength.ptp() > 0:
        sizes_all = s_min + (
            node_strength - node_strength.min()
        ) / node_strength.ptp() * (s_max - s_min)
    else:
        sizes_all = np.full(len(words), s_min)

    size_dict = {word: size for word, size in zip(words, sizes_all)}
    node_sizes = [size_dict.get(n, s_min) for n in G_sub.nodes()]

    # Edge widths based on co-occurrence strength
    edge_weights = np.array(
        [d["weight"] for _, _, d in G_sub.edges(data=True)], dtype=float
    )
    ew_min, ew_max = 0.1, 10
    if len(edge_weights) > 0 and edge_weights.ptp() > 0:
        edge_widths = ew_min + (
            edge_weights - edge_weights.min()
        ) / edge_weights.ptp() * (ew_max - ew_min)
    else:
        edge_widths = (
            np.full(len(edge_weights), ew_min) if len(edge_weights) > 0 else []
        )

    # Node colors by semantic axis
    node_colors = []
    for n in G_sub.nodes():
        cat = category_of.get(n, "unknown")
        axis = axis_of_category.get(cat, "unknown")
        color = axis_palette.get(axis, "gray")
        node_colors.append(color)

    # Layout and draw
    pos = nx.spring_layout(G_sub, seed=42, k=0.5)

    plt.figure(figsize=(10, 10))

    # Draw nodes
    nx.draw_networkx_nodes(
        G_sub,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.85,
        linewidths=1,
        edgecolors="navy",
    )

    # Draw edges (only if there are edges)
    if len(G_sub.edges) > 0:
        nx.draw_networkx_edges(
            G_sub, pos, width=edge_widths, alpha=0.7, edge_color="steelblue"
        )

    # Draw labels
    nx.draw_networkx_labels(
        G_sub, pos, font_size=9, font_weight="bold", font_color="black"
    )

    # Legend
    handles = [Patch(color=c, label=ax) for ax, c in axis_palette.items()]
    plt.legend(handles=handles, title="Axis category", loc="upper right")

    plt.title(title, fontsize=40)
    plt.axis("off")
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved graph: {output_path}")


if __name__ == "__main__":
    main()
