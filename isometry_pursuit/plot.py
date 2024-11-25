import matplotlib.pyplot as plt
import numpy as np


def box_plot(losses_1, losses_2, D, xlabels, ylabel, filename):

    data = np.asarray([np.asarray(losses_1) - D, np.asarray(losses_2) - D]).transpose()
    box = plt.boxplot(
        data,
        patch_artist=True,  # For custom box colors
        medianprops=dict(
            color="black", linewidth=2
        ),  # Set median line color and thickness
    )

    # Customize boxplot colors
    colors = ["turquoise", "pink"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    plt.xticks([1, 2], xlabels, fontsize=30)
    # plt.title('Side-by-Side Boxplot')
    plt.yscale("symlog")
    plt.ylim(0, 2 * data.max())
    plt.ylabel(ylabel, fontsize=30)
    # Add faint lines connecting individual data points
    for left, right in zip(data[:, 0], data[:, 1]):
        if right > left:
            color = "turquoise"
        elif left > right:
            color = "pink"  # Red for increasing loss, blue for decreasing
        else:
            color = "black"
        plt.plot([1, 2], [left, right], color=color, alpha=0.5, linewidth=2)

    plt.savefig(filename)
