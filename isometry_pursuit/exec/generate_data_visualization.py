#!/usr/bin/env python3
"""
Generate data visualization (data.png) showing random vectors representing data features.
This replaces the super_simple_example.ipynb notebook functionality.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow
from matplotlib.legend_handler import HandlerPatch

# Add parent directory to path to import isometry_pursuit modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class HandlerArrow(HandlerPatch):
    """Custom legend handler for the arrow"""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        arrow = FancyArrow(
            0,
            0.5 * height,
            width,
            0,
            length_includes_head=True,
            head_width=0.4 * height,
            head_length=0.4 * width,
            color=orig_handle.get_edgecolor(),
            linewidth=orig_handle.get_linewidth(),
        )
        return [arrow]


def generate_data_visualization(output_dir="figures", num_vectors=30, seed=2):
    """
    Generate data visualization showing random vectors representing data features.

    Parameters:
    -----------
    output_dir : str
        Directory to save the plot
    num_vectors : int
        Number of random vectors to generate
    seed : int
        Random seed for reproducibility
    """

    print("Generating data visualization...")

    # Set random seed
    np.random.seed(seed)

    # All vectors start at the origin
    start_x = np.zeros(num_vectors)
    start_y = np.zeros(num_vectors)

    # Generate random directions and magnitudes for the vectors
    angles = np.random.uniform(0, 2 * np.pi, num_vectors)  # Random angles
    magnitudes = np.exp(np.random.uniform(np.log(0.5), np.log(2), num_vectors))

    end_x = magnitudes * np.cos(angles)  # X components
    end_y = magnitudes * np.sin(angles)  # Y components

    # Add two vectors at right angles to each other
    theta = np.pi / 3
    end_x = np.append(end_x, [np.cos(theta), np.cos(theta + np.pi / 2)])
    end_y = np.append(end_y, [np.sin(theta), np.sin(theta + np.pi / 2)])
    start_x = np.append(start_x, [0, 0])
    start_y = np.append(start_y, [0, 0])

    xlim = 2
    ylim = 2

    # Define a bright Miami-inspired color gradient
    miami_cmap = LinearSegmentedColormap.from_list(
        "MiamiPalette", ["#FF1493", "#00FFFF", "#8A2BE2"]
    )
    colors = miami_cmap(np.linspace(0, 1, len(end_x)))

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each vector
    for i in range(len(end_x)):
        plt.quiver(
            start_x[i],
            start_y[i],  # Starting point (all at origin)
            end_x[i],
            end_y[i],  # Vector direction
            angles="xy",
            scale_units="xy",
            scale=1,  # Keep vectors at correct scale
            color=colors[i],  # Use continuous Miami-inspired colors
            alpha=0.9,  # Make colors pop
        )

    # Set axis limits and remove gridlines
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)

    # Set custom tick locations and labels
    tick_positions = [-1, 0, 1]
    plt.xticks(tick_positions)
    plt.yticks(tick_positions)
    ax.tick_params(labelsize=30)

    # Add a custom arrow in the legend
    arrow_for_legend = mpatches.FancyArrow(0, 0, 1, 0, color="black")
    plt.legend(
        handles=[arrow_for_legend],
        labels=[r"$X_{.p}$"],
        fontsize=40,
        frameon=False,
        handler_map={mpatches.FancyArrow: HandlerArrow()},
        loc="upper left",
    )

    # Display the plot
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "data.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Data visualization saved to {output_file}")


def generate_selection_visualization(output_dir="figures", num_vectors=30, seed=2):
    """
    Generate selection visualization showing random vectors with selected right-angle vectors.

    Parameters:
    -----------
    output_dir : str
        Directory to save the plot
    num_vectors : int
        Number of random vectors to generate
    seed : int
        Random seed for reproducibility
    """

    print("Generating selection visualization...")

    # Set random seed
    np.random.seed(seed)

    # All vectors start at the origin
    start_x = np.zeros(num_vectors)
    start_y = np.zeros(num_vectors)

    # Generate random directions and magnitudes for the vectors
    angles = np.random.uniform(0, 2 * np.pi, num_vectors)  # Random angles
    magnitudes = np.exp(np.random.uniform(np.log(0.5), np.log(2), num_vectors))

    end_x = magnitudes * np.cos(angles)  # X components
    end_y = magnitudes * np.sin(angles)  # Y components

    # Add two vectors at right angles to each other
    theta = np.pi / 3
    x1, y1 = np.cos(theta), np.sin(theta)
    x2, y2 = -y1, x1  # Perpendicular vector

    end_x = np.append(end_x, [x1, x2])
    end_y = np.append(end_y, [y1, y2])
    start_x = np.append(start_x, [0, 0])
    start_y = np.append(start_y, [0, 0])

    xlim = 2
    ylim = 2

    # Define a bright Miami-inspired color gradient
    miami_cmap = LinearSegmentedColormap.from_list(
        "MiamiPalette", ["#FF1493", "#00FFFF", "#8A2BE2"]
    )
    colors = miami_cmap(np.linspace(0, 1, len(end_x)))

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot random vectors with faint colors
    for i in range(num_vectors):
        plt.quiver(
            start_x[i],
            start_y[i],  # Starting point (all at origin)
            end_x[i],
            end_y[i],  # Vector direction
            angles="xy",
            scale_units="xy",
            scale=1,  # Keep vectors at correct scale
            color=[colors[i][0], colors[i][1], colors[i][2], 0.3],  # Make them faint
            alpha=0.3,
        )

    # Plot right-angle vectors with bold colors
    for i in range(num_vectors, len(end_x)):
        plt.quiver(
            start_x[i],
            start_y[i],  # Starting point (all at origin)
            end_x[i],
            end_y[i],  # Vector direction
            angles="xy",
            scale_units="xy",
            scale=1,  # Keep vectors at correct scale
            color=colors[i],  # Use prominent colors for right-angle vectors
            alpha=0.9,
        )

    # Add a right-angle symbol
    symbol_scale = 0.2  # Scale for the right-angle symbol
    corner_x = x1 * symbol_scale
    corner_y = y1 * symbol_scale

    symbol_x = [corner_x, corner_x + x2 * symbol_scale, x2 * symbol_scale]
    symbol_y = [corner_y, corner_y + y2 * symbol_scale, y2 * symbol_scale]

    plt.plot(symbol_x, symbol_y, color="black", linewidth=2.5)

    # Add a dotted black circle with radius 1
    from matplotlib.patches import Circle

    circle = Circle(
        (0, 0), 1, color="black", linestyle="dotted", fill=False, linewidth=2.5
    )
    ax.add_artist(circle)

    # Add a small line and label "r = 1" on the x-axis
    ax.text(1.05, 0.05, "r = 1", fontsize=40, color="black", ha="left", va="bottom")

    # Set axis limits
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)

    # Set custom tick locations and labels
    tick_positions = [-1, 0, 1]
    plt.xticks(tick_positions)
    plt.yticks(tick_positions)
    ax.tick_params(labelsize=30)

    # Create a dummy dotted circle for the legend
    dotted_circle = Circle((0, 0), radius=1.2, edgecolor="black", facecolor="none")

    # Custom legend handler for dotted circle
    class HandlerDottedCircle(HandlerPatch):
        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            circle = Circle(
                (width / 2, height / 2),
                radius=min(width, height) / 4,
                edgecolor=orig_handle.get_edgecolor(),
                facecolor="none",
                linestyle=":",
                linewidth=orig_handle.get_linewidth(),
            )
            circle.set_transform(trans)
            return [circle]

    # Add the legend with the custom handler
    plt.legend(
        handles=[dotted_circle],
        fontsize=40,
        frameon=False,
        labels=[r"$X_{.S}$ isometric"],
        handler_map={Circle: HandlerDottedCircle()},
        loc="upper left",
        handletextpad=0.3,  # Adjusts the space between the symbol and text
        labelspacing=0.0,  # Adjusts vertical space between legend entries
    )

    # Display the plot
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "selection.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Selection visualization saved to {output_file}")


def generate_normalization_visualization(output_dir="figures", num_vectors=30, seed=2):
    """
    Generate normalization visualization showing vectors after exponential transformation.

    Parameters:
    -----------
    output_dir : str
        Directory to save the plot
    num_vectors : int
        Number of random vectors to generate
    seed : int
        Random seed for reproducibility
    """

    print("Generating normalization visualization...")

    # Set random seed
    np.random.seed(seed)

    # All vectors start at the origin
    start_x = np.zeros(num_vectors)
    start_y = np.zeros(num_vectors)

    # Generate random directions and magnitudes for the vectors
    angles = np.random.uniform(0, 2 * np.pi, num_vectors)  # Random angles
    magnitudes = np.exp(np.random.uniform(np.log(0.5), np.log(2), num_vectors))

    end_x = magnitudes * np.cos(angles)  # X components
    end_y = magnitudes * np.sin(angles)  # Y components

    # Apply exponential transformation (this is the key difference)
    from isometry_pursuit.transformation import exponential_transformation

    Xs = exponential_transformation(np.asarray([end_x, end_y]))
    end_x = Xs[0]
    end_y = Xs[1]

    # Add two vectors at right angles to each other
    theta = np.pi / 3
    x1, y1 = np.cos(theta), np.sin(theta)
    x2, y2 = -y1, x1  # Perpendicular vector

    end_x = np.append(end_x, [x1, x2])
    end_y = np.append(end_y, [y1, y2])
    start_x = np.append(start_x, [0, 0])
    start_y = np.append(start_y, [0, 0])

    xlim = 2
    ylim = 2

    # Define a bright Miami-inspired color gradient
    miami_cmap = LinearSegmentedColormap.from_list(
        "MiamiPalette", ["#FF1493", "#00FFFF", "#8A2BE2"]
    )
    colors = miami_cmap(np.linspace(0, 1, len(end_x)))

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot random vectors with faint colors
    for i in range(num_vectors):
        plt.quiver(
            start_x[i],
            start_y[i],  # Starting point (all at origin)
            end_x[i],
            end_y[i],  # Vector direction
            angles="xy",
            scale_units="xy",
            scale=1,  # Keep vectors at correct scale
            color=[colors[i][0], colors[i][1], colors[i][2], 0.3],  # Make them faint
            alpha=0.3,
        )

    # Plot right-angle vectors with bold colors
    for i in range(num_vectors, len(end_x)):
        plt.quiver(
            start_x[i],
            start_y[i],  # Starting point (all at origin)
            end_x[i],
            end_y[i],  # Vector direction
            angles="xy",
            scale_units="xy",
            scale=1,  # Keep vectors at correct scale
            color=colors[i],  # Use prominent colors for right-angle vectors
            alpha=0.9,
        )

    # Add a right-angle symbol
    symbol_scale = 0.2  # Scale for the right-angle symbol
    corner_x = x1 * symbol_scale
    corner_y = y1 * symbol_scale

    symbol_x = [corner_x, corner_x + x2 * symbol_scale, x2 * symbol_scale]
    symbol_y = [corner_y, corner_y + y2 * symbol_scale, y2 * symbol_scale]

    plt.plot(symbol_x, symbol_y, color="black", linewidth=2.5)

    # Add a dotted black circle with radius 1
    from matplotlib.patches import Circle

    circle = Circle(
        (0, 0), 1, color="black", linestyle="dotted", fill=False, linewidth=2.5
    )
    ax.add_artist(circle)

    # Add a small line and label "r = 1" on the x-axis
    ax.text(1.05, 0.05, "r = 1", fontsize=40, color="black", ha="left", va="bottom")

    # Set axis limits
    plt.xlim(-xlim, xlim)
    plt.ylim(-ylim, ylim)

    # Set custom tick locations and labels
    tick_positions = [-1, 0, 1]
    plt.xticks(tick_positions)
    plt.yticks(tick_positions)
    ax.tick_params(labelsize=30)

    # Create a dummy dotted circle for the legend
    dotted_circle = Circle((0, 0), radius=1.2, edgecolor="black", facecolor="none")

    # Custom legend handler for dotted circle
    class HandlerDottedCircle(HandlerPatch):
        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            circle = Circle(
                (width / 2, height / 2),
                radius=min(width, height) / 4,
                edgecolor=orig_handle.get_edgecolor(),
                facecolor="none",
                linestyle=":",
                linewidth=orig_handle.get_linewidth(),
            )
            circle.set_transform(trans)
            return [circle]

    # Add the legend with the custom handler
    plt.legend(
        handles=[dotted_circle],
        fontsize=40,
        frameon=False,
        labels=[r"$X_{.S}$ isometric"],
        handler_map={Circle: HandlerDottedCircle()},
        loc="upper left",
        handletextpad=0.3,  # Adjusts the space between the symbol and text
        labelspacing=0.0,  # Adjusts vertical space between legend entries
    )

    # Display the plot
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "normalization.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Normalization visualization saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate data visualizations")
    parser.add_argument("--output-dir", default="figures", help="Output directory")
    parser.add_argument(
        "--num-vectors", type=int, default=30, help="Number of vectors to generate"
    )
    parser.add_argument("--seed", type=int, default=2, help="Random seed")

    args = parser.parse_args()

    # Generate all three visualizations
    generate_data_visualization(
        output_dir=args.output_dir, num_vectors=args.num_vectors, seed=args.seed
    )

    generate_selection_visualization(
        output_dir=args.output_dir, num_vectors=args.num_vectors, seed=args.seed
    )

    generate_normalization_visualization(
        output_dir=args.output_dir, num_vectors=args.num_vectors, seed=args.seed
    )


if __name__ == "__main__":
    main()
