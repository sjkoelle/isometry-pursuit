"""Plot strict pairwise optimizer defeat frequencies for synthetic and real data."""
import argparse
from pathlib import Path

from isometry_pursuit.swap_plot import plot_defeat_matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('results'))
    parser.add_argument('--output-dir', type=Path, default=Path('figures/optimizer_defeats'))
    parser.add_argument('--tie-tolerance', type=float, default=1e-9)
    parser.add_argument('--synthetic-bundle',type=Path,help='Complete validated 3250-matrix Lambda bundle, shown leftmost')
    parser.add_argument('--two-rows',action='store_true',help='Arrange Synthetic/Iris/Wine/Ethanol in a 2x2 panel')
    args = parser.parse_args()
    plot_defeat_matrices(args.results_dir, args.output_dir, args.tie_tolerance,args.synthetic_bundle,args.two_rows)


if __name__ == '__main__':
    main()
