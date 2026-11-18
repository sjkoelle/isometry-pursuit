"""Plot real-data regrets against the best observed search per replicate."""
import argparse
from pathlib import Path

from isometry_pursuit.swap_plot import plot_real_regrets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('figures/real_search_regrets'))
    args = parser.parse_args()
    plot_real_regrets(args.results_dir, args.output)


if __name__ == '__main__':
    main()
