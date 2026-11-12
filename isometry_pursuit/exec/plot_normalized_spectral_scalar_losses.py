"""Plot scalar profiles for normalized spectral losses."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use linear coordinates so the plot preserves the convexity of the
    # scalar profiles as functions of squared singular value u.
    u = np.linspace(0.2, 3.0, 1200)
    losses = {
        "Nuclear: $1-\\sqrt{u}$": 1.0 - np.sqrt(u),
        "Log determinant: $-\\frac{1}{2}\\log u$": -0.5 * np.log(u),
        "Frobenius inverse squared: $u^{-1}-1$": u**-1 - 1.0,
    }

    colors = ["#0072B2", "#E69F00", "#009E73"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for (label, values), color in zip(losses.items(), colors, strict=True):
        ax.plot(u, values, lw=2.4, color=color, label=label)

    ax.axvline(1.0, color="0.35", lw=1.2, ls="--")
    ax.axhline(0.0, color="0.65", lw=0.9)
    ax.annotate(
        "orthonormal value",
        xy=(1.0, 0.0),
        xytext=(1.13, 2.1),
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 1.0},
        color="0.25",
        fontsize=10,
    )

    ax.set_xlim(u.min(), u.max())
    ax.set_xlabel(r"Squared singular value $u=\sigma^2$")
    ax.set_ylabel(r"Scalar loss $f(\sqrt{u})$")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, loc="upper right")

    for suffix in ("pdf", "png"):
        path = output_dir / f"normalized_spectral_scalar_losses.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        print(path)


if __name__ == "__main__":
    main()
